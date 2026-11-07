"""Presentation of already drawn study figures, with no executable persistence.

Capture contexts retain one ordinary Matplotlib canvas while the caller renders
its group into a temporary directory. Reconstructing that group may calculate
display geometry (for example KDE curves); subsequent appearance edits only
change artist properties. Persist the JSON overrides, never a Figure or pickle.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import re
import threading

import numpy as np
from matplotlib.collections import Collection, PathCollection
from matplotlib.contour import ContourSet
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from ..core.figure_series import SeriesDescriptor, stable_series_id, validate_series_styles
from ..core.figure_style import resolve_series_color
from ..core.figure_style_mapping import project_legacy_style
from .matplotlib_targets import apply_text_overrides, remember_automatic_text
from .style import font_family_list


_local = threading.local()


@dataclass
class CapturedFigure:
    name: str
    figure: object = None
    data: object = None


@contextmanager
def capture_figure(name):
    """Retain an unstyled figure by filename stem; other canvases are cleared.

PNG writes from the two study save helpers are suppressed. Their callers may
still write numerical sidecars, so use a disposable output directory.
"""
    result = CapturedFigure(Path(str(name)).stem)
    previous = getattr(_local, 'capture', None)
    _local.capture = result
    try:
        yield result
        if result.figure is None:
            raise ValueError(f'The figure group did not produce {result.name!r}.')
    except BaseException:
        if result.figure is not None:
            result.figure.clear()
        raise
    finally:
        _local.capture = previous


@contextmanager
def styled_figures(overrides):
    """Apply JSON-shaped per-stem styling during ordinary figure regeneration."""
    previous = getattr(_local, 'styles', None)
    _local.styles = overrides or {}
    try:
        yield
    finally:
        _local.styles = previous


def capture_saved_figure(figure, name, source=None):
    """Save-hook protocol: True means the caller must skip PNG save and clear."""
    name = Path(str(name)).stem
    capture = getattr(_local, 'capture', None)
    if capture is not None:
        if name == capture.name:
            if capture.figure is not None:
                raise ValueError(f'Duplicate generated figure name: {name}')
            capture.figure = figure
            capture.data = None if source is None else source.copy(deep=True)
        else:
            figure.clear()
        return True
    overrides = (getattr(_local, 'styles', None) or {}).get(name)
    if overrides:
        StudyFigureStyle(figure).apply(**overrides)
    return False


def _rgb(value):
    return to_hex(to_rgba(value), keep_alpha=False)


def _replace_color(value, overrides):
    rgba = to_rgba(value)
    if rgba[3] == 0 or _rgb(rgba) not in overrides:
        return value
    replacement = to_rgba(overrides[_rgb(rgba)])
    return (*replacement[:3], rgba[3])


def bind_event_panel(axis, *, tag, event_id, start_idx, end_idx):
    """Bind annotations to the event actually drawn, independent of panel order.

    Include the automatic caption and displayed samples: an annotation quoting a
    measurement/population must be reviewed again if that scientific context
    changes. Only the short displayed traces are hashed, never a full recording.
    """
    if isinstance(event_id, np.generic):
        event_id = event_id.item()
    identity = dict(tag=str(tag), event_id=event_id, start_idx=int(start_idx),
                    end_idx=int(end_idx), caption=axis.get_title())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                      separators=(',', ':')).encode())
    for line in axis.lines:
        values = np.asarray(line.get_xydata(), dtype='<f8')
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    axis._nanoscope_event_panel_identity = digest.hexdigest()


class StudyFigureStyle:
    """Reset-safe appearance edits for the original artists of one canvas.

Per-panel labels are keyed by decimal axis index. Event titles instead follow a
stable event/trace/caption identity; legacy ordinal titles cannot relabel events.
Colour overrides are keyed by
the original RGB colour, preserving alpha and linking matching legend handles.
Continuous colour maps, numerical coordinates, limits and scales are preserved.
Single data/fit-colour fields are intentionally unused: use colour groups to
retain categorical identities on the multi-panel scientific figures.
"""

    def __init__(self, figure):
        self.figure = figure
        figure.canvas.draw()
        self._defaults = []
        self._axes = list(figure.axes)
        self._original_suptitle = figure._suptitle
        remember_automatic_text(figure)
        self._axis_defaults = [(ax, tuple(ax.get_xlim()), tuple(ax.get_ylim()),
                                copy(ax.xaxis.get_major_locator()), copy(ax.yaxis.get_major_locator()))
                               for ax in self._axes]
        self._grid_defaults = [(ax, [v.get_visible() for v in ax.get_xgridlines()],
                               [v.get_visible() for v in ax.get_ygridlines()]) for ax in self._axes]
        self._texts = list(figure.findobj(Text))
        self._lines, self._collections, self._patches = [], [], []
        self._legends = [*figure.legends, *(ax.get_legend() for ax in self._axes if ax.get_legend())]
        self._panel_defaults = [dict(key=str(index), title=ax.get_title(),
                                    x_label=ax.get_xlabel(), y_label=ax.get_ylabel())
                                for index, ax in enumerate(self._axes)]
        for descriptor, ax in zip(self._panel_defaults, self._axes):
            if hasattr(ax, '_nanoscope_event_panel_identity'):
                descriptor['event_identity'] = ax._nanoscope_event_panel_identity
        self._remember(figure, ('facecolor', 'edgecolor'))
        for text in self._texts:
            self._remember(text, ('text', 'fontfamily', 'fontsize', 'fontweight', 'fontstyle',
                                  'horizontalalignment', 'parse_math', 'color', 'visible'))
        for ax in self._axes:
            self._remember(ax, ('facecolor',))
            for spine in ax.spines.values():
                self._remember(spine, ('edgecolor', 'linewidth', 'visible'))
            for line in [*ax.get_xgridlines(), *ax.get_ygridlines(),
                         *ax.xaxis.get_ticklines(), *ax.yaxis.get_ticklines()]:
                self._remember(line, ('color', 'linewidth', 'visible', 'markeredgecolor', 'alpha'))
            self._lines.extend(ax.lines)
            self._collections.extend(ax.collections)
            self._patches.extend(ax.patches)
        for legend in self._legends:
            self._remember(legend, ('visible',))
            self._remember(legend.get_frame(), ('facecolor', 'edgecolor'))
            # Composite legend entries (notably error bars) contain several
            # marker/cap/line artists. ``legend_handles`` exposes only one of
            # those, so traversing the drawing is required for linked colours.
            # The decorative frame is not a data series and must stay separate.
            for handle in legend.findobj(lambda artist: isinstance(artist, (Line2D, Collection, Patch))):
                if handle is legend.get_frame():
                    continue
                if isinstance(handle, Line2D):
                    self._lines.append(handle)
                elif isinstance(handle, Patch):
                    self._patches.append(handle)
                elif isinstance(handle, Collection):
                    self._collections.append(handle)
        self._lines = list(dict.fromkeys(self._lines))
        self._collections = list(dict.fromkeys(self._collections))
        self._patches = list(dict.fromkeys(self._patches))
        for line in self._lines:
            self._remember(line, ('color', 'linewidth', 'markersize', 'markerfacecolor', 'markeredgecolor', 'visible'))
        for collection in self._collections:
            fields = ['facecolors', 'edgecolors', 'linewidths', 'visible']
            if isinstance(collection, PathCollection):
                fields.append('sizes')
            self._remember(collection, fields)
        for patch in self._patches:
            self._remember(patch, ('facecolor', 'edgecolor', 'linewidth', 'visible'))
        self._groups = self._discover_colors()
        self._artist_groups = {}
        for artist in [*self._lines, *self._collections, *self._patches]:
            if isinstance(artist, Line2D):
                values = [artist.get_color()]
            elif isinstance(artist, Collection):
                values = ([*artist.get_facecolors(), *artist.get_edgecolors()]
                          if self._categorical(artist) else [])
            else:
                values = [artist.get_facecolor(), artist.get_edgecolor()]
            self._artist_groups[artist] = frozenset(_rgb(value) for value in values
                                                    if to_rgba(value)[3] != 0)
        self._series = self._describe_series()
        self._tag_series_artists()
        self._series_colors = {}

    def _remember(self, artist, fields):
        values = {field: deepcopy(getattr(artist, 'get_' + field)()) for field in fields}
        if isinstance(artist, Collection):
            # get_facecolors() resolves 'none' to an empty array and get_edgecolors()
            # resolves 'face'/mapped edges to RGBA. Feeding those arrays back to a
            # scalar-mapped Collection changes which surface receives its colours:
            # an unfilled ContourSet becomes filled on the next draw. Retain the
            # native input sentinels, not just their last rendered RGBA values.
            for field, attribute in (('facecolors', '_original_facecolor'),
                                     ('edgecolors', '_original_edgecolor')):
                if field not in values or not hasattr(artist, attribute):
                    continue
                original = getattr(artist, attribute)
                if isinstance(original, str) or artist.get_array() is not None:
                    values[field] = deepcopy(original)
        self._defaults.append((artist, values))

    def panels(self):
        return deepcopy(self._panel_defaults)

    def color_groups(self):
        return deepcopy(self._groups)

    def canonical_style(self, style):
        """Move applicable legacy edits into the shared element stores, once.

        Unavailable event identities and unrecognized entries remain untouched. A
        legacy ordinal event title is never promoted onto another event.
        """
        from .matplotlib_targets import panel_identity, storage_key
        value = deepcopy(style or {})
        appearance = value.setdefault('appearance', {})
        text = appearance.setdefault('text', {})
        panels = value.setdefault('panels', {})
        events = value.get('event_titles', {})
        for index, (axis, descriptor) in enumerate(zip(self._axes, self._panel_defaults)):
            old = panels.get(str(index), {})
            if not isinstance(old, dict):
                continue  # The normal validator will report malformed saved state.
            panel = panel_identity(axis, index)
            for field, key in (('title', 'title'), ('x_label', 'axis.bottom.label'),
                               ('y_label', 'axis.left.label')):
                owner = old
                name = field
                if field == 'title' and 'event_identity' in descriptor:
                    owner, name = events, descriptor['event_identity']
                if name not in owner or not isinstance(owner[name], str):
                    continue
                wording = owner.pop(name)
                if wording:
                    text.setdefault(storage_key(panel, key), {}).setdefault('text', wording)
            if str(index) in panels and not old:
                panels.pop(str(index))
        if not text:
            appearance.pop('text', None)
        colors = value.setdefault('colors', {})
        series = appearance.setdefault('series_styles', {})
        for descriptor in self._series:
            aliases = [key for key in colors if str(key).lower() == descriptor.identity_key]
            if aliases:
                try:
                    ink = to_hex(colors[aliases[-1]])  # same case-insensitive, named-color vocabulary as apply
                except (TypeError, ValueError):
                    continue  # Retain invalid legacy values for normal validation.
                series.setdefault(descriptor.series_id, {}).setdefault('color', ink)
                for key in aliases:
                    colors.pop(key)
        if not series:
            appearance.pop('series_styles', None)
        return value

    def _describe_series(self):
        result = []
        for group in self._groups:
            key = group['key']
            artists = [artist for artist, keys in self._artist_groups.items() if key in keys]
            # A multicolour collection/patch can be recoloured by original RGB,
            # but hiding/resizing the whole artist would affect another group.
            independent = bool(artists) and all(self._artist_groups[a] == {key} for a in artists)
            capabilities = ['color']
            if independent:
                capabilities.extend(('line_width_pt', 'visible'))
                if any(isinstance(a, PathCollection) or
                       (isinstance(a, Line2D) and a.get_marker() not in (None, 'None', '', ' '))
                       for a in artists):
                    capabilities.insert(2, 'marker_size_pt')
            result.append(SeriesDescriptor(stable_series_id('study', 'data', key), group['label'],
                identity_kind='artist', identity_key=key, color=key, capabilities=tuple(capabilities)))
        return tuple(result)

    def _tag_series_artists(self):
        def key(value):
            return stable_series_id('study', 'data', _rgb(value)) if to_rgba(value)[3] else ''
        for artist, groups in self._artist_groups.items():
            if len(groups) == 1:
                artist._nanoscope_series_id = stable_series_id('study', 'data', next(iter(groups)))
            elif isinstance(artist, PathCollection) and self._categorical(artist):
                colors = artist.get_facecolors()
                if not len(colors): colors = artist.get_edgecolors()
                if len(colors):
                    artist._nanoscope_series_point_ids = tuple(key(colors[i % len(colors)])
                        for i in range(len(artist.get_offsets())))
            elif isinstance(artist, Collection) and self._categorical(artist):
                colors = artist.get_edgecolors()
                if len(colors): artist._nanoscope_series_path_ids = tuple(key(color) for color in colors)
            elif isinstance(artist, Patch):
                color = artist.get_facecolor()
                if not to_rgba(color)[3]: color = artist.get_edgecolor()
                artist._nanoscope_series_id = key(color)

    def series_descriptors(self):
        """Retained original identities plus current effective ink for Copy format."""
        result = []
        for descriptor in self._series:
            artists = [artist for artist, keys in self._artist_groups.items()
                       if descriptor.identity_key in keys]
            values = {'color': self._series_colors.get(descriptor.identity_key, descriptor.color)}
            if 'line_width_pt' in descriptor.capabilities:
                width = float(np.asarray(artists[0].get_linewidth()).ravel()[0])
                if width > 0: values['line_width_pt'] = width
            if 'marker_size_pt' in descriptor.capabilities:
                for artist in artists:
                    if isinstance(artist, PathCollection) and len(artist.get_sizes()):
                        values['marker_size_pt'] = float(np.sqrt(artist.get_sizes()[0])); break
                    if isinstance(artist, Line2D) and artist.get_marker() not in (None, 'None', '', ' '):
                        values['marker_size_pt'] = float(artist.get_markersize()); break
            result.append(replace(descriptor, extra={'effective_style': values}))
        return tuple(result)

    def _apply_series_properties(self, overrides):
        descriptors = {entry.identity_key: entry for entry in self._series}
        for artist, keys in self._artist_groups.items():
            if len(keys) != 1:
                continue
            descriptor = descriptors[next(iter(keys))]
            state = overrides.get(descriptor.series_id, {})
            if 'visible' in descriptor.capabilities and 'visible' in state:
                artist.set_visible(state['visible'])
            if 'line_width_pt' in descriptor.capabilities and 'line_width_pt' in state:
                artist.set_linewidth(state['line_width_pt'])
            if 'marker_size_pt' in descriptor.capabilities and 'marker_size_pt' in state:
                if isinstance(artist, PathCollection):
                    artist.set_sizes([state['marker_size_pt'] ** 2])
                elif isinstance(artist, Line2D):
                    artist.set_markersize(state['marker_size_pt'])
        for legend in self._legends:
            for handle, text in zip(legend.legend_handles, legend.get_texts()):
                keys = self._artist_groups.get(handle, ())
                if len(keys) == 1:
                    descriptor = descriptors[next(iter(keys))]
                    if ('visible' in descriptor.capabilities and
                            overrides.get(descriptor.series_id, {}).get('visible') is False):
                        text.set_visible(False)

    @staticmethod
    def _categorical(collection):
        # Scalar-mapped artists (including colourbars) keep their colour scale.
        # Explicit fixed-colour line contours also carry scalar level indices,
        # but their ink is a categorical identity, not a continuous colour scale.
        return (collection.get_array() is None or
                isinstance(collection, ContourSet) and not collection.filled
                and collection.colors is not None)

    def _discover_colors(self):
        groups = {}
        def add(value, label=''):
            rgba = to_rgba(value)
            if rgba[3] == 0:
                return
            key = _rgb(rgba)
            entry = groups.setdefault(key, dict(key=key, color=key, label=key))
            if label and not str(label).startswith('_'):
                entry['label'] = str(label)
        for line in self._lines:
            add(line.get_color(), line.get_label())
        for patch in self._patches:
            add(patch.get_facecolor(), patch.get_label())
            add(patch.get_edgecolor(), patch.get_label())
        for collection in self._collections:
            if self._categorical(collection):
                for color in [*collection.get_facecolors(), *collection.get_edgecolors()]:
                    add(color, collection.get_label())
        return list(groups.values())

    def apply(self, appearance=None, panels=None, colors=None, event_titles=None,
              style_binding=None, **_carried_extensions):
        """Apply validated appearance to original properties, without accumulating."""
        state = appearance.to_dict() if hasattr(appearance, 'to_dict') else dict(appearance or {})
        if style_binding is not None:
            state = project_legacy_style(style_binding, 'study_figure_style',
                                         {'appearance': state})['appearance']
        panels = dict(panels or {})
        event_titles = dict(event_titles or {})
        colors = {str(key).lower(): to_hex(value) for key, value in dict(colors or {}).items()}
        # Validate before touching the canvas, making malformed saved edits atomic.
        if any(not re.fullmatch(r'#[0-9a-f]{6}', key) for key in colors):
            raise ValueError('Colour override keys must identify original RGB colours.')
        for key, panel in panels.items():
            if not isinstance(key, str) or not re.fullmatch(r'0|[1-9][0-9]*', key) or not isinstance(panel, dict):
                raise ValueError('A panel override requires a decimal panel index.')
            if not panel.keys() <= {'title', 'x_label', 'y_label'} or any(not isinstance(v, str) for v in panel.values()):
                raise ValueError('Panel overrides must contain text labels only.')
        if any(not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{64}', key)
               or not isinstance(value, str) for key, value in event_titles.items()):
            raise ValueError('Event titles require a saved event identity and text.')
        # Curation can remove a category or shrink an event gallery. Keep the
        # caller's complete saved overrides intact, applying only present artists.
        known = {entry['key'] for entry in self._groups}
        colors = {key: value for key, value in colors.items() if key in known}
        series = validate_series_styles(state.get('series_styles') or {})
        # Original RGB identity wins over inherited palette defaults. An explicit
        # Series edit (including Apply palette) wins over the older colour editor.
        colors = {entry.identity_key: resolve_series_color(style_binding, index=index,
                  identity_color=colors.get(entry.identity_key, entry.color),
                  explicit_color=series.get(entry.series_id, {}).get('color'))
                  for index, entry in enumerate(self._series)}
        # A no-op mapping must not rewrite native representations during Reset.
        colors = {key: value for key, value in colors.items() if _rgb(value) != key}
        for key, low, high in (('title_size', 6, 48), ('axis_size', 6, 36), ('tick_size', 6, 30),
                               ('legend_size', 6, 30), ('line_width', .25, 10), ('marker_size', 1, 30)):
            if key in state and (type(state[key]) not in (int, float) or not np.isfinite(state[key])
                                 or (state[key] <= 0 if style_binding is not None else not low <= state[key] <= high)):
                raise ValueError(f'Invalid figure appearance: {key}')
        for key in ('title', 'x_label', 'y_label', 'font_family'):
            if key in state and not isinstance(state[key], str):
                raise ValueError(f'Invalid figure appearance: {key}')
        if self._original_suptitle is None and self.figure._suptitle is not None:
            self.figure._suptitle.remove()
            self.figure._suptitle = None
        for ax, xlim, ylim, xlocator, ylocator in self._axis_defaults:
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.xaxis.set_major_locator(copy(xlocator))
            ax.yaxis.set_major_locator(copy(ylocator))
        for ax, xs, ys in self._grid_defaults:
            ax.xaxis.grid(bool(xs and xs[0]))
            ax.yaxis.grid(bool(ys and ys[0]))
        for artist, values in self._defaults:
            artist.set(**deepcopy(values))
            if isinstance(artist, Collection):
                # Resolve the restored mapping before an explicit colour edit.
                # This updates ink only; array, levels, norm and cmap stay native.
                artist.update_scalarmappable()
        dark = state.get('background') == 'dark'
        foreground, background = ('#eeeeee', '#17191d') if dark else ('#202226', '#ffffff')
        if state.get('background') in ('dark', 'light'):
            self.figure.set_facecolor(background)
            for ax in self._axes:
                ax.set_facecolor(background)
                for spine in ax.spines.values():
                    spine.set_edgecolor(foreground)
                ax.tick_params(colors=foreground)
            for text in self._texts:
                text.set_color(foreground)
            for legend in self._legends:
                legend.get_frame().set_facecolor(background)
                legend.get_frame().set_edgecolor(foreground)
        # The chosen family's stack -- Arial's when none is chosen -- in every Text (and the
        # title below): a bare name is a warning on every draw where that family is not
        # installed, and the canvas item draws this figure outside any style.
        family = font_family_list(state.get('font_family'))
        for text in self._texts:
            text.set_fontfamily(family)
        for ax, descriptor in zip(self._axes, self._panel_defaults):
            panel = dict(panels.get(descriptor['key'], {}))
            if 'event_identity' in descriptor:
                # Gallery panel positions change after curation. Never treat a
                # legacy positional title as approval for the current event.
                panel.pop('title', None)
                identity = descriptor['event_identity']
                if identity in event_titles:
                    panel['title'] = event_titles[identity]
            # An empty override keeps the original label and a blank one hides it, as in
            # every plot window (design 13.4 Q1).
            for field, setter in (('title', ax.set_title), ('x_label', ax.set_xlabel), ('y_label', ax.set_ylabel)):
                if panel.get(field):
                    setter(panel[field].strip()).set_parse_math(False)
            if 'axis_size' in state:
                ax.xaxis.label.set_fontsize(state['axis_size'])
                ax.yaxis.label.set_fontsize(state['axis_size'])
            if 'tick_size' in state:
                for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.offsetText, ax.yaxis.offsetText]:
                    text.set_fontsize(state['tick_size'])
            if 'title_size' in state:
                ax.title.set_fontsize(state['title_size'])
            if isinstance(state.get('frame'), bool) and ax.get_visible() and ax.axison:
                # Keep scientific axes/ticks/limits and hidden density marginals intact.
                for side in ('top', 'right'):
                    ax.spines[side].set_visible(state['frame'])
            if isinstance(state.get('grid'), bool):
                if state['grid']: ax.grid(True, alpha=.2)
                else: ax.grid(False)
        if state.get('title'):
            self.figure.suptitle(state['title'].strip()).set_parse_math(False)
        if self.figure._suptitle is not None:
            if 'title_size' in state:
                self.figure._suptitle.set_fontsize(state['title_size'])
            self.figure._suptitle.set_fontfamily(family)
            if state.get('background') in ('dark', 'light'):
                self.figure._suptitle.set_color(foreground)
        for legend in self._legends:
            if isinstance(state.get('legend_visible'), bool):
                legend.set_visible(state['legend_visible'])
            if 'legend_size' in state:
                for text in legend.get_texts():
                    text.set_fontsize(state['legend_size'])
        for line in self._lines:
            for field in ('color', 'markerfacecolor', 'markeredgecolor'):
                getattr(line, 'set_' + field)(_replace_color(getattr(line, 'get_' + field)(), colors))
            if 'line_width' in state:
                line.set_linewidth(state['line_width'])
            if 'marker_size' in state:
                line.set_markersize(state['marker_size'])
        for patch in self._patches:
            patch.set_facecolor(_replace_color(patch.get_facecolor(), colors))
            patch.set_edgecolor(_replace_color(patch.get_edgecolor(), colors))
            if 'line_width' in state:
                patch.set_linewidth(state['line_width'])
        for collection in self._collections:
            if self._categorical(collection):
                for field in ('facecolors', 'edgecolors'):
                    values = getattr(collection, 'get_' + field)()
                    # No-op assignments can erase 'none'/'face' mapping semantics.
                    # In particular, never turn an empty contour face into a
                    # scalar-mapped face merely to recolour its edge.
                    if any(to_rgba(value)[3] and _rgb(value) in colors for value in values):
                        getattr(collection, 'set_' + field)([_replace_color(v, colors) for v in values])
            if 'line_width' in state:
                collection.set_linewidth(state['line_width'])
            if 'marker_size' in state and isinstance(collection, PathCollection):
                collection.set_sizes([state['marker_size'] ** 2])
        self._apply_series_properties(series)
        self._series_colors = colors
        apply_text_overrides(self.figure, state, binding=style_binding)
        return self.figure
