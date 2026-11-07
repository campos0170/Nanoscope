"""Serial, immutable figure exports with an explicit result for every planned file.

The coordinator accepts a capture callback, never an analysis/update callback. Each
capture is released before another figure is materialized. Dialogs belong to the
caller; filenames and existing-file identities are fixed before overwrite approval.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
import shutil
import unicodedata

from ..core.atomic_write import staged_path
from ..core.figure_style import FigureStyle
from .figure_bundle import FigureSnapshot, MAX_SNAPSHOT_BYTES, write_figure_bundle, write_frozen_figure_image
from .figure_export_style import snapshot_export_style, style_from_binding
from .plot_export_options import resolve_export_options
from ..plot.pg_svg import SVG_POINT_WARN_THRESHOLD


@dataclass(frozen=True)
class ExportFile:
    path: Path
    extension: str
    original_identity: object


@dataclass(frozen=True)
class ExportTarget:
    figure_id: str
    name: str
    revision: object
    files: tuple[ExportFile, ...]
    width_mm: float | None
    snapshot: FigureSnapshot | None = None
    unavailable: str = ''
    height_mm: float | None = None


@dataclass(frozen=True)
class BatchExportPlan:
    targets: tuple[ExportTarget, ...]
    sizing: str = 'native'
    dpi: int = 300

    @property
    def note(self):
        return ('Resizing scales all artwork, including fonts. Column widths come from each '
                'captured style; reference widths are not publication requirements. '
                'Native exports preserve the captured look. No analysis is run.')

    @property
    def paths(self):
        return tuple(file.path for target in self.targets for file in target.files)


@dataclass(frozen=True)
class BatchExportResult:
    figure_id: str
    name: str
    path: Path
    status: str
    error: str = ''


def _file_identity(path):
    path = Path(path)
    try:
        entry = path.lstat()
    except FileNotFoundError:
        return None
    try:
        target = path.stat()
    except FileNotFoundError:
        return ('dangling', entry.st_dev, entry.st_ino, entry.st_mtime_ns)
    return (str(path.resolve()), entry.st_dev, entry.st_ino, entry.st_mtime_ns,
            target.st_dev, target.st_ino, target.st_size, target.st_mtime_ns)


def _stem(name):
    value = unicodedata.normalize('NFC', str(name))
    value = re.sub(r'\s*[·/]\s*', ' - ', value)
    value = re.sub(r'[^\w .-]', '_', value).strip(' .') or 'Figure'
    while len(value.encode('utf-8')) > 180:
        value = value[:-1]
    return value


def _width(style, sizing):
    if sizing == 'native':
        return None
    if sizing == 'style':
        return float(style.look['canvas_width_mm']) if style is not None and 'canvas_width_mm' in style.look else None
    key = {'single': 'single_mm', 'double': 'double_mm'}[sizing]
    if style is None or key not in style.columns:
        raise ValueError(f'This figure has no {sizing}-column width in its captured style; use Native.')
    return float(style.columns[key])


def build_export_plan(owner, selected_ids, directory, *, formats=('pdf',), sizing='native',
                      dpi=300, snapshots=None, ready=None):
    """Freeze names/revisions/paths before dialogs; snapshots map ID to (name, snapshot)."""
    directory = Path(directory).absolute()
    if not directory.is_dir():
        raise ValueError('Choose an existing export folder.')
    formats = tuple(dict.fromkeys('.' + str(value).lower().lstrip('.') for value in formats))
    if not formats or not set(formats) <= {'.pdf', '.svg', '.png', '.zip'}:
        raise ValueError('Choose PNG, PDF, SVG, or a figure bundle.')
    if sizing not in ('native', 'style', 'single', 'double'):
        raise ValueError('Choose the style canvas default, Native, Single column, or Double column sizing.')
    resolve_export_options((1, 1), dpi=dpi)  # Validate without a renderer.
    ready = ready or getattr(owner, 'export_ready', lambda key: (True, ''))
    targets, used = [], set()
    snapshots = dict(snapshots or {})
    ids = tuple(dict.fromkeys(selected_ids))
    if set(ids) & snapshots.keys():
        raise ValueError('Live figures and snapshots must have distinct identities.')
    for key in (*ids, *snapshots):
        snapshot = None
        reason = ''
        token = None
        style = None
        if key in snapshots:
            name, snapshot = snapshots[key]
            if not isinstance(snapshot, FigureSnapshot):
                raise TypeError('Saved selections must provide immutable FigureSnapshots.')
            try:
                style = snapshot_export_style(snapshot)
            except ValueError as exc:
                reason = str(exc)
        else:
            record = owner.get(key)
            name = record.spec.name if record is not None else 'Unavailable figure'
            allowed, reason = ready(key)
            if record is None:
                reason = 'This live figure is no longer available.'
            elif not allowed:
                reason = reason or 'This figure is busy.'
            elif not owner.checkpoint(key):
                reason = 'The current figure edits could not be preserved.'
            else:
                record = owner.get(key)
                token = owner.revision_token(key)
                if record.spec.result.status == 'empty':
                    reason = 'This figure has no committed result. Update it before exporting.'
                raw = record.spec.presentation.style_snapshot
                if raw.get('style_id'):
                    try:
                        style = style_from_binding({'version': 1, 'ref': getattr(record.spec.presentation, 'style_ref', None) or
                            {'style_id': raw['style_id'], 'revision': raw['revision']}, 'snapshot': raw,
                            'overrides': getattr(record.spec.presentation, 'overrides', {})})
                    except ValueError as exc:
                        reason = str(exc)
        width, height = None, None
        if not reason:
            try:
                width = _width(style, sizing)
                if sizing == 'style' and style is not None and 'canvas_height_mm' in style.look:
                    height = float(style.look['canvas_height_mm'])
            except ValueError as exc:
                reason = str(exc)
        stem, candidate, number = _stem(name), _stem(name), 1
        while candidate.casefold() in used:
            number += 1
            candidate = f'{stem} ({number})'
        used.add(candidate.casefold())
        files = tuple(ExportFile(directory / (candidate + ext), ext,
                                 _file_identity(directory / (candidate + ext))) for ext in formats)
        targets.append(ExportTarget(str(key), str(name), token, files, width, snapshot, reason, height))
    if not targets:
        raise ValueError('Select at least one live figure or saved snapshot.')
    return BatchExportPlan(tuple(targets), sizing, int(dpi))


def _snapshot_bytes(snapshot):
    _meta, blobs = snapshot.to_storage()
    total = sum(len(value) for value in blobs.values())
    if total > MAX_SNAPSHOT_BYTES:
        raise ValueError('The captured figure exceeds the snapshot resource budget.')
    return total


def _require_capture_budget(page, maximum):
    from PySide6 import QtWidgets
    from ..plot.pg_svg import count_vector_points
    ordered = getattr(page, 'figure_export_items', None)
    target = getattr(page, 'figure_export_target', None)
    items = list(ordered()) if callable(ordered) else [target()] if callable(target) else []
    roots = []
    for item in items:
        if isinstance(item, QtWidgets.QGraphicsScene):
            roots.extend(child for child in item.items() if child.parentItem() is None)
        else:
            roots.append(item)
    points = sum(count_vector_points(item) for item in roots)
    if points >= maximum:
        raise ValueError(f'This figure draws {points:,} individual vector markers, reaching the '
                         f'batch limit of {maximum:,}. Nothing was sampled or exported. '
                         'Use individual Export to review the large vector capture.')


def run_export_plan(plan, *, owner, capture, is_current, ready=None, cancelled=None,
                    progress=None, overwrite=False, saved_is_current=None,
                    vector_point_limit=SVG_POINT_WARN_THRESHOLD):
    """Write serially; return written/failed/cancelled per file, never inferred success.

    ``capture(id)`` returns a FigureSnapshot of the committed drawing. ``is_current``
    guards the project owner/epoch. ``ready(id)`` rejects busy native surfaces. The
    caller may pump events in progress/cancel callbacks; guards run again afterwards.
    """
    if type(vector_point_limit) is not int or vector_point_limit <= 0:
        raise ValueError('The reviewed vector point limit must be a positive integer.')
    ready = ready or getattr(owner, 'export_ready', lambda key: (True, ''))
    cancelled = cancelled or (lambda: False)
    saved_is_current = saved_is_current or (lambda key: True)
    previous = owner.current_id() if owner is not None else None
    previous_resident = previous is not None and owner.page(previous) is not None
    selected = owner.selected_ids() if owner is not None else ()
    results = []
    cancellation_seen = False

    def report(target, file, status, error=''):
        result = BatchExportResult(target.figure_id, target.name, file.path, status, str(error))
        results.append(result)
        if progress is not None:
            try:
                progress(result, len(results), len(plan.paths))
            except Exception as exc:
                # A failed status widget cannot turn a file already committed into a
                # second, fictional failed export. Preserve its real outcome and note it.
                results[-1] = replace(result, error='; '.join(filter(None, (
                    result.error, f'Progress could not be displayed: {exc}'))))

    def restore_previous():
        if previous is None:
            clear = getattr(owner, 'clear_current', None)
            if callable(clear): clear()
            return True
        if owner.get(previous) is None:
            return True
        if previous_resident:
            return bool(owner.activate(previous))
        restore = getattr(owner, 'restore_navigation', None)
        if callable(restore):
            return bool(restore(previous))
        # Compatibility for external coordinators without the navigation-only API.
        return bool(owner.activate(previous) and owner.suspend(previous))

    def still_owned(target):
        if not is_current():
            raise ValueError('The project changed; remaining files were not exported.')
        if target.snapshot is not None and not saved_is_current(target.figure_id):
            raise ValueError('The saved figure was deleted or replaced; nothing was exported for it.')
        if target.snapshot is None:
            token = owner.revision_token(target.figure_id)
            if token is None or token[:2] != target.revision[:2]:
                raise ValueError('The figure was deleted or replaced; nothing was exported for it.')

    try:
        for target in plan.targets:
            cancellation_seen = cancellation_seen or cancelled()
            if cancellation_seen:
                for file in target.files:
                    report(target, file, 'cancelled', 'Export cancelled before this figure.')
                continue
            snapshot = target.snapshot
            dormant = False
            capture_error = None
            try:
                if target.unavailable:
                    raise ValueError(target.unavailable)
                still_owned(target)
                if snapshot is None:
                    if owner.revision_token(target.figure_id) != target.revision:
                        raise ValueError('The figure changed after export planning; select it again.')
                    allowed, reason = ready(target.figure_id)
                    if not allowed:
                        raise ValueError(reason or 'This figure is busy.')
                    dormant = owner.page(target.figure_id) is None
                    if dormant and not owner.activate(target.figure_id):
                        raise ValueError('This saved live figure could not be opened for capture.')
                    _require_capture_budget(owner.page(target.figure_id), vector_point_limit)
                    snapshot = capture(target.figure_id)
                    if not isinstance(snapshot, FigureSnapshot):
                        raise TypeError('Figure capture did not return an immutable snapshot.')
                    still_owned(target)
                size = _snapshot_bytes(snapshot)
            except Exception as exc:
                capture_error = exc
            finally:
                if dormant and is_current():
                    cleanup = []
                    # Even a failed restoration must still attempt temporary-page release.
                    try:
                        if not restore_previous():
                            cleanup.append('The original figure view could not be restored.')
                    except Exception as exc:
                        cleanup.append(str(exc))
                    try:
                        if not owner.suspend(target.figure_id):
                            cleanup.append('The temporary figure could not be released after capture.')
                    except Exception as exc:
                        cleanup.append(str(exc))
                    owner.set_selected(selected)
                    if cleanup:
                        capture_error = capture_error or ValueError(' '.join(cleanup))
            if capture_error is not None:
                for file in target.files:
                    report(target, file, 'failed', capture_error)
                continue
            for file in target.files:
                try:
                    still_owned(target)
                    if file.original_identity is not None and not overwrite:
                        raise FileExistsError('This file already exists; replacement was not approved.')
                    if _file_identity(file.path) != file.original_identity:
                        raise ValueError('The destination changed after export planning; it was left untouched.')
                    width, height = target.width_mm, target.height_mm
                    if width is None and height is not None:
                        width = height * snapshot.width / snapshot.height
                    if height is None and width is not None:
                        height = width * snapshot.height / snapshot.width
                    options = None if file.extension == '.zip' else resolve_export_options(
                        (snapshot.width, snapshot.height), extension=file.extension,
                        native=width is None and height is None, width=width, height=height, dpi=plan.dpi)
                    required = max(8 * 1024 * 1024, 2 * size)
                    if options is not None and file.extension == '.png' and not options.native:
                        required = max(required, options.width * options.height * 4)
                    if shutil.disk_usage(file.path.parent).free < required:
                        raise OSError('There is not enough free space to stage this export safely.')
                    with staged_path(file.path) as staged:
                        if file.extension == '.zip':
                            write_figure_bundle(staged, snapshot)
                        else:
                            write_frozen_figure_image(staged, snapshot, options)
                        if not staged.is_file() or staged.stat().st_size == 0:
                            raise ValueError('The export writer did not create a file.')
                        still_owned(target)
                        if _file_identity(file.path) != file.original_identity:
                            raise ValueError('The destination changed during export; it was left untouched.')
                    report(target, file, 'written')
                except Exception as exc:
                    report(target, file, 'failed', exc)
    finally:
        if owner is not None and is_current():
            restore_previous()
            owner.set_selected(selected)
    return tuple(results)
