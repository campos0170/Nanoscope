"""Explicit surface seams for figures that can leave and re-enter the resident view cache.

These adapters own no widgets. The registry owns page construction, source binding, adoption
and retirement; native workspace codecs still own scientific snapshot capture/restoration.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math

from ..core.figure_history_state import HistoryCapsule
from ..core.figure_spec import FigureValidationError, freeze_json, thaw_json

# Only these reviewed scalar/tuple memos cross a page lifetime. Never copy a page's __dict__.
_MEMOS = {
    "distributions": ("_plot_armed", "_stale", "_stale_reason", "_stale_reasons",
                      "_applied_kde_overlay", "_stale_drawn"),
    "iv": (),
    "voltage_series": ("_drawn_metric", "_drawn_abs", "_fit_words_for", "_applied_fit"),
    "distribution_summaries": ("_words_for", "_stale"),
    "comparison_figures": ("_words_for", "_stale", "_fit_pending"),
    "event_waveforms": ("_view_shown", "_density_shown", "_density_logs",
                        "_normalize_shown", "_align_shown"),
    "noise": ("_view_shown", "_asd_shown"),
    "subpopulations": ("_embedding_shown",),
}
_TUPLE_MEMOS = frozenset({"_words_for", "_fit_words_for", "_density_shown", "_density_logs",
                         "_reset_took", "_stale_drawn"})
_NATIVE = {
    "iv": ("capture_iv", "restore_iv"),
    "voltage_series": ("capture_voltage_dependence", "restore_voltage_dependence"),
    "distribution_summaries": ("capture_distribution_summaries", "restore_distribution_summaries"),
    "comparison_figures": ("capture_comparison_figures", "restore_comparison_figures"),
    "event_waveforms": ("capture_event_plots", "restore_event_plots"),
    "noise": ("capture_psd", "restore_psd"),
    "subpopulations": ("capture_subpopulation", "restore_subpopulation"),
}


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


def _plain(value):
    return thaw_json(freeze_json(value))


def _plots(page, kind):
    """All actual plots, including hidden views, keyed independently of Python addresses."""
    if kind == "distributions":
        return {"main": page._plot, "residuals": page._resid}
    if kind == "comparison_figures":
        return {}
    if kind == "event_waveforms":
        return {"density": page._density_main, "ensemble": page._ensemble_plot}
    if kind == "noise":
        return {"psd": page._plot, "fft": page._fft_plot,
                "spectrogram": page._spec_plot, "allpoints": page._hist_plot}
    if kind == "subpopulations":
        return {"embedding": page._embed_view._plot.getPlotItem(),
                "waveforms": page._wave_view._plot.getPlotItem()}
    return {"main": page._plot}


def _capture_views(page, kind):
    views = {}
    for key, plot in _plots(page, kind).items():
        view = plot.getViewBox()
        views[key] = {"range": [[float(v) for v in pair] for pair in view.viewRange()],
                      "auto": list(view.state["autoRange"])}
    canvas = page._view_state() if kind == "comparison_figures" else None
    return _plain({"plots": views, "canvas": canvas})


def _validate_views(page, kind, raw):
    if not isinstance(raw, Mapping) or not isinstance(raw.get("plots"), Mapping):
        raise FigureValidationError("Invalid figure viewport capsule")
    plots = _plots(page, kind)
    if set(raw["plots"]) != set(plots):
        raise FigureValidationError("Figure viewport identities changed")
    for value in raw["plots"].values():
        if not isinstance(value, Mapping):
            raise FigureValidationError("Invalid plot viewport")
        ranges, auto = value.get("range"), value.get("auto")
        if (not isinstance(ranges, (list, tuple)) or len(ranges) != 2
                or not isinstance(auto, (list, tuple)) or len(auto) != 2):
            raise FigureValidationError("Invalid viewport axes")
        for pair in ranges:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
                    or pair[0] >= pair[1]):
                raise FigureValidationError("Invalid viewport range")
        if any(type(v) is not bool and (type(v) not in (int, float) or not 0 <= v <= 1)
               for v in auto):
            raise FigureValidationError("Invalid viewport autorange")
    if kind == "comparison_figures":
        page._validate_view(thaw_json(raw.get("canvas")))
    elif raw.get("canvas") is not None:
        raise FigureValidationError("Unexpected figure canvas state")


def _restore_views(page, kind, raw):
    for key, plot in _plots(page, kind).items():
        value = raw["plots"][key]
        view = plot.getViewBox()
        # A view whose X is linked to another (the Distributions Residual strip follows the
        # histogram) is given its saved Y only. pyqtgraph passes a linked view's new X range on to
        # the view it is linked to, lined up by SCREEN geometry, so the strip, restored after the
        # histogram, would hand the histogram the X it was saved with in the layout of the window
        # that saved it, and a typed range came back about 1 % of its width off. The strip's X is
        # its partner's. Its own X auto-range flag (pyqtgraph's A button on the strip can turn it
        # on, and a file may hold it) is not replayed: the histogram's saved X already is what the
        # strip framed. No restored view is linked on Y (the Event map's marginal plots are, but
        # they are not among ``_plots``; see test_p2_figure_surface_adapters), so a view's Y is
        # always its own.
        follows_x = view.linkedView(0) is not None
        view.setRange(xRange=None if follows_x else value["range"][0], yRange=value["range"][1],
                      padding=0, disableAutoRange=True)
        for axis, enabled in enumerate(value["auto"]):
            if not (axis == 0 and follows_x):
                view.enableAutoRange(axis=axis, enable=enabled)
    if kind == "comparison_figures":
        page._restore_view(thaw_json(raw["canvas"]))


class NeedsResidentHistory(RuntimeError):
    """A configuration Undo must use its live calculation/validation path."""


class FigureViewAdapter:
    """Non-owning protocol implementation shared by the native and Plots adapters."""

    def __init__(self, page, kind):
        self.page, self.kind = page, kind

    def capture_record(self, previous):
        from ..core.figure_codecs import merge_native_capture
        record = merge_native_capture(self.capture_native(), previous)
        views = {**thaw_json(record.spec.view.canvas_view),
                 "surface": _capture_views(self.page, self.kind)}
        return replace(record, spec=replace(record.spec, view=replace(
            record.spec.view, canvas_view=views)))

    def restore_record(self, record, *, mode="project"):
        from ..core.figure_codecs import encode_legacy
        if mode not in ("project", "resume") or record.spec.kind != self.kind:
            raise FigureValidationError("Incompatible figure restore request")
        artifact = encode_legacy(record.spec, record.resources, materialize_defaults=True)
        self.page._figure_suspended = False
        surface = record.spec.view.canvas_view.get("surface")
        if surface is not None:
            _validate_views(self.page, self.kind, surface)
        report = self.restore_native(artifact)
        if not report.skipped and surface is not None:
            _restore_views(self.page, self.kind, surface)
        return report

    def _controllers(self):
        from .figure_text import figure_text_for
        return [controller for plot in _plots(self.page, self.kind).values()
                if (controller := figure_text_for(plot)) is not None]

    def _owned_focus(self):
        from PySide6 import QtWidgets
        focus = QtWidgets.QApplication.focusWidget()
        if focus is None:
            return None
        if self.page.isAncestorOf(focus):
            return focus
        toolbar = getattr(focus.window(), "toolbar", None)
        if callable(toolbar):
            bar = toolbar()
            adapter = bar.adapter() if hasattr(bar, "adapter") else None
            if adapter is not None and adapter.page is self.page:
                return focus
        return None

    def _commit_edits(self):
        from .figure_text import _alive
        focus = self._owned_focus()
        if focus is not None:
            valid = getattr(focus, "hasAcceptableInput", None)
            if callable(valid) and not valid():
                raise FigureValidationError("Finish the incomplete figure control value")
            focus.clearFocus()  # the same commit route as leaving this edit by hand
        for controller in self._controllers():
            editor = controller.editor
            if _alive(editor) and not editor._done:
                editor._finish()
        self.page._appearance_widget._close_burst()

    def can_suspend(self):
        from .figure_text import _alive
        if getattr(self.page, "_gather_worker", None) is not None:
            return False, "A figure calculation is still running"
        focus = self._owned_focus()
        valid = getattr(focus, "hasAcceptableInput", None)
        if callable(valid) and not valid():
            return False, "Finish the incomplete figure control value"
        for controller in self._controllers():
            if _alive(controller.popover) and controller.popover.isVisible():
                return False, "Finish the open figure properties editor"
        return True, ""

    def suspend(self):
        allowed, reason = self.can_suspend()
        if not allowed:
            raise RuntimeError(reason)
        self._commit_edits()
        self.page._figure_suspended = True
        fitter = getattr(self.page, "_nonlinear_fitter", None)
        dialog = getattr(fitter, "dialog", None)
        if dialog is not None:
            # The resident page can be reused for a different figure with identical
            # source rows. A fit dialog belongs to the figure that opened it.
            dialog.reject()
        burst = self.page._appearance_widget._burst
        if burst is not None:
            burst.stop()
        for controller in self._controllers():
            controller.close_editors()

    def _capture_memos(self):
        page = self.page
        names = _MEMOS[self.kind] + ("_reset_said", "_reset_took")
        return _plain({name: getattr(page, name) for name in names if hasattr(page, name)})

    def _install_memos(self, raw):
        for name, value in raw.items():
            setattr(self.page, name, _tuples(value) if name in _TUPLE_MEMOS else value)

    def _capture_history(self):
        return self.page._appearance_widget.capture_history_capsule()

    def _install_history(self, capsule):
        from .figure_quantity_native import migrate_native_history
        from ..core import analysis_workspace
        artifact = getattr(analysis_workspace, _NATIVE[self.kind][0])(self.page)
        capsule = migrate_native_history(capsule, artifact.payload, self.kind)
        self.page._appearance_widget.restore_history_capsule(capsule)

    def capture_transient(self) -> HistoryCapsule:
        self._commit_edits()
        widget = self.page._appearance_widget
        history = self._capture_history()
        if history is None:
            raise FigureValidationError("Native figure is missing its appearance history")
        status = self.page.statusBar() if hasattr(self.page, "statusBar") else None
        transient = {"version": 1, "kind": self.kind, "memos": self._capture_memos(),
                     "views": _capture_views(self.page, self.kind),
                     "appearance_reset": {"said": widget._reset_said,
                                          "took": widget._reset_took},
                     "status": status.currentMessage() if status is not None else ""}
        return replace(history, extra={**thaw_json(history.extra), "surface": transient})

    def install_transient(self, capsule):
        if not isinstance(capsule, HistoryCapsule):
            capsule = HistoryCapsule.from_dict(capsule)
        surface = thaw_json(capsule.extra.get("surface"))
        if surface is None:
            # A project-loaded, still-dormant figure may have received a Style edit.
            # Frozen restoration established quantity anchors; only its history needs installing.
            self._install_history(capsule)
            return
        if (not isinstance(surface, dict) or surface.get("version") != 1
                or surface.get("kind") != self.kind or not isinstance(surface.get("memos"), dict)
                or not isinstance(surface.get("appearance_reset"), dict)
                or not isinstance(surface.get("status"), str)):
            raise FigureValidationError("Incompatible figure surface capsule")
        permitted = set(_MEMOS[self.kind]) | {"_reset_said", "_reset_took"}
        if self.kind == "distributions":
            permitted.add("_identity")
        if not set(surface["memos"]) <= permitted:
            raise FigureValidationError("Unknown figure quantity memo")
        _validate_views(self.page, self.kind, surface["views"])
        widget = self.page._appearance_widget
        self._install_history(capsule)
        self._install_memos(surface["memos"])
        widget._reset_said = surface["appearance_reset"]["said"]
        widget._reset_took = _tuples(surface["appearance_reset"]["took"])
        widget._reset_bar = self.page.statusBar() if hasattr(self.page, "statusBar") else None
        _restore_views(self.page, self.kind, surface["views"])
        if widget._reset_bar is not None:
            widget._reset_bar.showMessage(surface["status"], 10000)
        signal = getattr(self.page, "pendingChanged", None)
        if signal is not None:
            signal.emit()


class NativeFigureAdapter(FigureViewAdapter):
    def __init__(self, page, kind):
        if kind not in _NATIVE:
            raise FigureValidationError(f"Unsupported native figure kind: {kind}")
        super().__init__(page, kind)

    def capture_native(self):
        from ..core import analysis_workspace
        self._commit_edits()
        return getattr(analysis_workspace, _NATIVE[self.kind][0])(self.page)

    def restore_native(self, artifact):
        from ..core import analysis_workspace
        # Validate/re-express all display pins before a renderer clears its old picture.
        artifact = _with_native_appearance(artifact, self.kind, _native_appearance(artifact, self.kind))
        return getattr(analysis_workspace, _NATIVE[self.kind][1])(self.page, artifact)


def adapter_for(page, kind) -> FigureViewAdapter:
    if kind == "distributions":
        from .figure_adapter_plots import PlotsFigureAdapter
        return PlotsFigureAdapter(page)
    return NativeFigureAdapter(page, kind)


def _preserve_view(record, previous):
    return replace(record, spec=replace(record.spec, view=replace(record.spec.view,
        canvas_view={**thaw_json(previous.spec.view.canvas_view),
                     **thaw_json(record.spec.view.canvas_view)})))


def _appearance_path(kind):
    return ("config", "appearance") if kind in ("voltage_series", "event_waveforms", "noise") else ("appearance",)


def _native_appearance(artifact, kind):
    value = artifact.payload
    for key in _appearance_path(kind):
        value = value.get(key, {})
    from .plot_appearance import PlotAppearance
    from .figure_quantity_native import migrate_native_appearance
    if kind in ("distribution_summaries", "comparison_figures"):
        value = {**value, "source_appearance": value.get("source_appearance",
                 artifact.payload.get("source_appearance", {}))}
    return PlotAppearance.from_mapping(migrate_native_appearance(value, artifact.payload, kind))


def _with_native_appearance(artifact, kind, appearance):
    payload = _plain(artifact.payload)
    value = payload
    path = _appearance_path(kind)
    for key in path[:-1]:
        value = value.setdefault(key, {})
    value[path[-1]] = appearance.to_dict()
    if kind in ("distribution_summaries", "comparison_figures") and appearance.source_appearance is not None:
        payload["source_appearance"] = thaw_json(appearance.source_appearance)
    return replace(artifact, payload=payload)


def apply_look_to_record(record, look, transient, *, keep_hand=True):
    """Apply one dormant appearance step without constructing a renderer or widgets."""
    if record.spec.kind == "distributions":
        from .figure_adapter_plots import apply_look_to_record as apply_plots
        return apply_plots(record, look, transient, keep_hand=keep_hand)
    from ..core.figure_codecs import encode_legacy, merge_native_capture
    if not isinstance(transient, HistoryCapsule):
        transient = HistoryCapsule.from_dict(transient)
    artifact = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    before = _native_appearance(artifact, record.spec.kind)
    from .figure_quantity_native import migrate_native_history
    transient = migrate_native_history(transient, artifact.payload, record.spec.kind)
    if transient.codec != "nanoscope.appearance-history/1" or thaw_json(transient.base) != before.to_dict():
        raise FigureValidationError("Dormant appearance history disagrees with the figure")
    after = before.restyled(look, keep_hand=keep_hand,
        vocabulary="matplotlib_appearance" if record.spec.kind == "comparison_figures" else "plot_appearance")
    if after == before:
        return record, transient
    result = merge_native_capture(_with_native_appearance(artifact, record.spec.kind, after), record)
    return _preserve_view(result, record), transient.note(after.to_dict())


def history_state(capsule):
    """Read the same bounded undo/redo availability while the figure has no live widgets."""
    from .plot_workflow import HistoryState
    if not isinstance(capsule, HistoryCapsule):
        capsule = HistoryCapsule.from_dict(capsule)
    back, forward = len(capsule.past), len(capsule.future)
    grain = "configuration" if capsule.codec == "nanoscope.plot-config-history/1" else "appearance"
    suffix = f" ({capsule.dropped} oldest steps dropped)" if capsule.dropped else ""
    return HistoryState(bool(back), bool(forward),
                        f"Undo {grain} ({back} steps){suffix}" if back else f"Nothing to undo{suffix}",
                        f"Redo {grain} ({forward} steps)" if forward else "Nothing to redo",
                        back + capsule.dropped)


def move_history(record, capsule, direction="undo"):
    """Move dormant history values without applying scientific calculations."""
    if direction not in ("undo", "redo"):
        raise FigureValidationError("History direction must be undo or redo")
    if record.spec.kind == "distributions":
        from .figure_adapter_plots import move_history as move_plots
        return move_plots(record, capsule, direction)
    from ..core.figure_codecs import encode_legacy, merge_native_capture
    from .plot_appearance import PlotAppearance
    if not isinstance(capsule, HistoryCapsule):
        capsule = HistoryCapsule.from_dict(capsule)
    artifact = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    before = _native_appearance(artifact, record.spec.kind)
    from .figure_quantity_native import migrate_native_history
    capsule = migrate_native_history(capsule, artifact.payload, record.spec.kind)
    if capsule.codec != "nanoscope.appearance-history/1" or thaw_json(capsule.base) != before.to_dict():
        raise FigureValidationError("Dormant appearance history disagrees with the figure")
    moved = capsule.undo() if direction == "undo" else capsule.redo()
    if moved is capsule:
        return record, capsule
    after = PlotAppearance.from_mapping(thaw_json(moved.base))
    result = merge_native_capture(_with_native_appearance(artifact, record.spec.kind, after), record)
    extra = thaw_json(moved.extra)
    surface = extra.get("surface")
    if surface is not None:
        # Match the live appearance widget's reset-sentence lapse; undo changes words/pins.
        from .quantity_reset import typed_words, pinned_axes, stands
        reset = surface["appearance_reset"]
        took = reset["took"]
        if took is not None:
            names, axes, primary = took
            if not stands(names, typed_words(after) + pinned_axes(after, axes, primary)):
                if surface["status"] == reset["said"]:
                    surface["status"] = ""
                reset.update(said="", took=None)
                if "_reset_said" in surface["memos"]:
                    surface["memos"]["_reset_said"] = ""
    return _preserve_view(result, record), replace(moved, extra=extra)


def configure_source_binding(page, source_group, follow, on_changed):
    """Keep one visible source-policy control and one callback per live page."""
    from PySide6 import QtCore, QtWidgets
    box = getattr(page, "_figure_follow_check", None)
    if box is None:
        box = QtWidgets.QCheckBox("Follow selection", source_group)
        box.setObjectName("figureFollowSelection")
        box.setToolTip("Use the selected recording for this figure. Off keeps its current source.")
        box.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        layout = source_group.layout()
        if isinstance(layout, QtWidgets.QFormLayout):
            layout.addRow(box)
        else:
            layout.addWidget(box)
        page._figure_follow_check = box
        box.show()  # A new child added after its parent is visible starts hidden in Qt.
    previous = getattr(page, "_figure_follow_callback", None)
    if previous is not None:
        box.toggled.disconnect(previous)
    with QtCore.QSignalBlocker(box):
        box.setChecked(bool(follow))
    page._figure_follow_callback = on_changed
    if on_changed is not None:
        box.toggled.connect(on_changed)


def initial_history(record) -> HistoryCapsule:
    """A loaded record starts from its saved values, without constructing a page."""
    from ..core.figure_codecs import encode_legacy
    artifact = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    if record.spec.kind == "distributions":
        from .figure_adapter_plots import _native_config
        from .plots_panel import plot_config_to_json, _HISTORY_DEPTH
        return HistoryCapsule("nanoscope.plot-config-history/1",
            plot_config_to_json(_native_config(artifact)), extra={"depth": _HISTORY_DEPTH,
            "origin": 0, "past_origins": [], "future_origins": [], "opened": 0,
            "norm_refusal_records": [], "config_loss_records": []})
    return HistoryCapsule("nanoscope.appearance-history/1",
                          _native_appearance(artifact, record.spec.kind).to_dict())


def capture_applied_record(record):
    """Return the applied snapshot with its current presentation and original resources."""
    from ..core.figure_codecs import encode_legacy, merge_native_capture
    artifact = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    payload = _plain(artifact.payload)
    kind = record.spec.kind
    if kind == "distributions":
        from .figure_adapter_plots import _native_config, _with_native_config
        from .plots_panel import plot_config_to_json, plot_config_from_json
        render = payload.get("phase2_render")
        if render is not None:
            applied = render.get("applied_config")
            if not applied:
                raise FigureValidationError("This figure has no applied plot to snapshot")
            current = plot_config_to_json(_native_config(artifact))
            shown = {**current, **applied, "style": current["style"],
                     "axes": {**current["axes"], **applied.get("axes", {})}}
            artifact = _with_native_config(artifact, plot_config_from_json(shown))
            payload = _plain(artifact.payload)
            payload["selected_sources"] = render["drawn_sources"]
            payload["phase2_render"]["gate"] = {"armed": True, "stale": False,
                                                   "reason": "", "reasons": []}
    elif kind in ("distribution_summaries", "comparison_figures"):
        if not payload.get("applied"):
            raise FigureValidationError("This figure has no applied analysis to snapshot")
        payload["draft"] = _plain(payload["applied"])
        payload["stale"] = False
    elif kind == "voltage_series":
        config = payload.get("config", {})
        if "applied_fit" in config:
            config["selected_fit"] = config["applied_fit"]
    elif kind in ("noise", "event_waveforms"):
        # A frozen snapshot describes the result already drawn. The live record
        # retains the unapplied draft, which belongs only to its Update workflow.
        payload.pop("pending_analysis", None)
    return _preserve_view(merge_native_capture(replace(artifact, payload=payload), record), record)


def capture_applied_spec(record):
    return capture_applied_record(record).spec
