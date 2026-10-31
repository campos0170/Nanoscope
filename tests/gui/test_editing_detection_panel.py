"""DetectionPanel: what the editing foundation needs from the Detection form.

Design ``docs/EDITING_FOUNDATION_DESIGN.md`` 5.3, 3.8 and 3.2, item D:

* A FREQUENCY THE USER SET SURVIVES A NYQUIST CLAMP. Leaving a recording clamps the
  low-pass, high-pass and notch boxes to the next recording's Nyquist (it has to: a
  cutoff the kernel refuses must not be runnable). On ``27ec0eb`` the clamp was also
  where the value STAYED: a 20 kHz low-pass typed on a 50 kHz recording read 12499.9 Hz
  after a look at a 25 kHz one and went on reading it back on the 50 kHz one, where the
  next run would have filtered at the clamp. The panel now remembers what each box was
  set to (by the user or ``set_params``) and puts it back wherever the range can hold
  it; a value it cannot hold is announced, by ``clampNotice``, when a run there would
  read that box. Writing back a snapshot taken at the clamp (an Undo, a re-seed) keeps
  what was set.
* DELETING A PROTOCOL OR A SAVED ML PRESET ASKS FIRST. Neither library is reached by
  Edit ▸ Undo (3.8), and both deletions were one click with nothing asked.
* THE SIGNALS THE UNDO WIRING LABELS ITS STEPS WITH: ``protocolRecalled`` (once per
  recall, after the form settles), ``mlParamsChanged`` (an ML edit, a seed, a preset
  load -- each once) and ``mlPresetLoaded``.
* THE ML FORM IS A PIPELINE WHATEVER ITS NOTCH BOX HOLDS. The form's snapshot reads it at
  every seed, leave and save, so an entry that made ``ml_params()`` raise stopped a switch
  of recording from finishing.

Synthetic data only: the two "recordings" are rates handed to ``set_recording``, and the
two window tests build synthetic documents under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QInputDialog, QMessageBox

from nanoscope.analysis.conditioning_recipes import complete_recipe, recipes_equal
from nanoscope.analysis.ml_preprocess import DEFAULT_PRESET
from nanoscope.core.document import TraceDocument
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.panels import DetectionPanel

FAST, SLOW = 50_000.0, 25_000.0          # Nyquist 25 kHz and 12.5 kHz
HOST_GUEST = "Host-guest (CB7 + adamantyl)"


@pytest.fixture(autouse=True)
def _clean_stores():
    """Both user libraries start and end empty (QSettings is redirected by conftest, but
    it is process-wide, so one test's protocol would be in the next one's combo)."""
    def _clear():
        s = QSettings()
        s.remove(DetectionPanel._PROTO_SETTINGS_KEY)
        s.remove(DetectionPanel._ML_PRESET_SETTINGS_KEY)
        s.sync()
    _clear()
    yield
    _clear()


@pytest.fixture
def panel(qtbot):
    p = DetectionPanel()
    qtbot.addWidget(p)
    return p


def _type(qtbot, box, text: str) -> None:
    """Type ``text`` into a spin box and commit it with Return, as a user does."""
    box.selectAll()
    qtbot.keyClicks(box, text)
    qtbot.keyClick(box, Qt.Key_Return)


def _named(monkeypatch, *names) -> None:
    """Answer the next QInputDialog.getText calls with ``names``, in order."""
    queue = list(names)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: (queue.pop(0), True)))


# ================================================================================================
# a frequency the user set survives a Nyquist clamp
# ================================================================================================

def test_a_typed_low_pass_comes_back_on_the_faster_recording(panel, qtbot):
    """The owner's case: 20 kHz typed at 50 kHz, a look at a 25 kHz recording, and back.

    On ``27ec0eb`` the box read 12499.9 Hz on the way back -- a value nobody typed, on a
    recording where 20 kHz is legal, with nothing on screen saying so."""
    panel.set_recording(FAST)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    assert panel.filter_cutoff_hz.value() == pytest.approx(20_000.0)

    panel.set_recording(SLOW)
    assert panel.filter_cutoff_hz.value() < SLOW / 2      # still clamped: it cannot run

    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(20_000.0)


def test_the_clamp_is_announced_and_a_clean_arrival_says_nothing_is(panel, qtbot):
    notices: list[str] = []
    panel.clampNotice.connect(notices.append)
    panel.set_recording(FAST)
    assert notices == [""]
    panel.filter_enabled.setChecked(True)
    _type(qtbot, panel.filter_cutoff_hz, "20000")

    panel.set_recording(SLOW)
    assert notices[-1] == ("Low-pass 20 kHz is above this recording's Nyquist (12.5 kHz); "
                           "the form now reads 12.4999 kHz.")
    assert panel.clamp_notice() == notices[-1]

    panel.set_recording(FAST)
    assert notices[-1] == ""
    assert len(notices) == 3                               # once per set_recording


def test_several_clamps_are_one_sentence_naming_each(panel, qtbot):
    panel.set_recording(FAST)
    for stage in (panel.filter_enabled, panel.hp_enabled, panel.notch_enabled):
        stage.setChecked(True)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    _type(qtbot, panel.hp_cutoff_hz, "15000")
    _type(qtbot, panel.notch_freq_hz, "13000")
    notices: list[str] = []
    panel.clampNotice.connect(notices.append)

    panel.set_recording(SLOW)
    # The high-pass box holds two decimals, so it stops at 12499.99 Hz -- printed in
    # full, because '12.5 kHz' would read as the Nyquist it is clamped BELOW.
    assert notices == [
        "Low-pass 20 kHz, high-pass 15 kHz and notch 13 kHz are above this recording's "
        "Nyquist (12.5 kHz); the form now reads 12.4999 kHz, 12.49999 kHz and "
        "12.4999 kHz."]

    panel.set_recording(FAST)
    assert (panel.filter_cutoff_hz.value(), panel.hp_cutoff_hz.value(),
            panel.notch_freq_hz.value()) == pytest.approx((20_000.0, 15_000.0, 13_000.0))


def test_a_run_uses_what_the_boxes_read_not_what_was_typed(panel, qtbot):
    """params() did not change meaning: the remembered value is for putting BACK, never
    for running at a rate that cannot hold it."""
    panel.set_recording(FAST)
    panel.filter_enabled.setChecked(True)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    panel.set_recording(SLOW)
    assert panel.params()["filter_cutoff_hz"] == pytest.approx(panel.filter_cutoff_hz.value())
    assert panel.params()["filter_cutoff_hz"] < SLOW / 2


def test_set_params_resets_what_each_box_was_set_to(panel, qtbot):
    """A seed or a recall replaces the remembered value, both ways: a value set below
    the clamp is not overruled by an older typed one, and a value set ABOVE it (a
    protocol saved on a faster recording) is what a faster recording gets back."""
    panel.set_recording(FAST)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    panel.set_params({"filter_cutoff_hz": 8_000.0})
    panel.set_recording(SLOW)
    assert panel.filter_cutoff_hz.value() == pytest.approx(8_000.0)
    assert panel.clamp_notice() == ""
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(8_000.0)

    panel.set_recording(SLOW)
    panel.set_params({"filter_cutoff_hz": 20_000.0, "hp_cutoff_hz": 30.0})
    assert panel.filter_cutoff_hz.value() < SLOW / 2
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(20_000.0)
    assert panel.hp_cutoff_hz.value() == pytest.approx(30.0)


def test_writing_back_a_snapshot_taken_at_the_clamp_keeps_what_was_set(panel, qtbot):
    """Design 3.2: a form step's snapshot is params(), written back through set_params.
    On the slower recording that snapshot holds the CLAMP, and writing it back as a set
    value -- an Undo of any step there, here an unrelated k_in edit, or a re-seed from a
    run made at the clamp -- would trade the typed 20 kHz for 12499.9 Hz: the notice
    would go quiet and the faster recording get the clamp back. So the ceiling a clamped
    box already reads sets nothing; any other value is still set as asked."""
    panel.set_recording(FAST)
    panel.filter_enabled.setChecked(True)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    panel.set_recording(SLOW)
    before = dict(panel.params())                           # the k_in step's "before"
    _type(qtbot, panel.k_in, "6")
    panel.set_params(before)                                # its Undo
    assert panel.k_in.value() == pytest.approx(before["k_in"])
    assert panel.clamp_notice().startswith("Low-pass 20 kHz is above")
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(20_000.0)

    panel.set_recording(SLOW)
    panel.set_params({"filter_cutoff_hz": 15_000.0})        # not the ceiling: set
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(15_000.0)


def test_an_untouched_panel_announces_nothing_and_its_built_values_come_back(panel):
    """The 10 kHz the low-pass box is built with is nobody's choice, and announcing it
    would be a false alarm on every arrival at 20 kHz or less, with the low-pass off or
    on. It still comes back where it fits, and once someone sets it, it is announced."""
    notices: list[str] = []
    panel.clampNotice.connect(notices.append)
    assert not panel.filter_enabled.isChecked()
    panel.set_recording(10_000.0)                           # Nyquist 5 kHz
    panel.filter_enabled.setChecked(True)
    panel.set_recording(10_000.0)
    assert notices == ["", ""]
    assert panel.filter_cutoff_hz.value() < 5_000.0
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(10_000.0)

    panel.set_params({"filter_cutoff_hz": 10_000.0})        # the same number, now set
    panel.set_recording(10_000.0)
    assert notices[-1].startswith("Low-pass 10 kHz is above")


def test_a_stage_a_run_does_not_read_is_not_announced(panel, qtbot):
    """An off stage reads no cutoff, and a notch spelled as explicit lines reads no
    frequency box: a run would miss nothing there. What was set still comes back."""
    panel.set_recording(FAST)
    _type(qtbot, panel.filter_cutoff_hz, "20000")           # set, the low-pass left off
    panel.notch_enabled.setChecked(True)
    _type(qtbot, panel.notch_freq_hz, "13000")
    panel.set_notch_lines([60.0, 120.0])                    # the lines are in force
    notices: list[str] = []
    panel.clampNotice.connect(notices.append)

    panel.set_recording(SLOW)
    assert notices == [""]
    panel.filter_enabled.setChecked(True)
    panel.set_notch_lines(None)                             # the series spelling again
    assert panel.clamp_notice() == (
        "Low-pass 20 kHz and notch 13 kHz are above this recording's Nyquist (12.5 kHz); "
        "the form now reads 12.4999 kHz and 12.4999 kHz.")
    panel.set_recording(FAST)
    assert (panel.filter_cutoff_hz.value(),
            panel.notch_freq_hz.value()) == pytest.approx((20_000.0, 13_000.0))


def test_a_value_this_recording_cannot_hold_reads_its_ceiling_whichever_way_it_came(panel):
    """What a box reads follows from what was set and this recording alone. 30 kHz set
    on a 100 kHz recording reads 24999.9 Hz on a 50 kHz one reached directly; reached by
    way of a 25 kHz one it kept that recording's 12499.9 Hz (on 27ec0eb as well), so the
    notice would say the form "now reads 12.4999 kHz" under a 25 kHz Nyquist."""
    panel.set_recording(100_000.0)
    panel.set_params({"filter_cutoff_hz": 30_000.0})
    panel.set_recording(FAST)
    direct = panel.filter_cutoff_hz.value()
    panel.set_recording(SLOW)
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == direct == pytest.approx(24_999.9)
    assert panel.clamp_notice() == ("Low-pass 30 kHz is above this recording's Nyquist "
                                    "(25 kHz); the form now reads 24.9999 kHz.")


def test_an_off_low_pass_keeps_its_box_value_as_what_was_set(panel, qtbot):
    panel.set_recording(FAST)
    panel.filter_enabled.setChecked(True)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    panel.set_params({"filter_cutoff_hz": None})           # the stage was off
    assert not panel.filter_enabled.isChecked()
    assert panel.filter_cutoff_hz.value() == pytest.approx(20_000.0)

    panel.set_recording(SLOW)
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(20_000.0)
    assert panel.params()["filter_cutoff_hz"] is None


@pytest.mark.parametrize("how", ["typed", "set_params"])
def test_a_listener_that_answers_a_change_with_another_value_has_the_last_word(
        panel, qtbot, how):
    """What was set is remembered BEFORE anyone else hears of it -- the recorder is the
    first listener to a box, and `_set_frequency` remembers before its box moves -- so a
    re-seed that answers a change by setting another value is not overwritten by the
    change it answered. Remembered after the box moved, a set_params would overrule the
    answer: the box would read 9 kHz and a faster recording get 20 kHz back."""
    panel.set_recording(FAST)

    def reseed(params):
        if params["filter_cutoff_hz"] is not None and params["filter_cutoff_hz"] > 15_000:
            panel.set_params({"filter_cutoff_hz": 9_000.0})

    panel.filter_enabled.setChecked(True)
    panel.paramsChanged.connect(reseed)
    if how == "typed":
        _type(qtbot, panel.filter_cutoff_hz, "20000")
    else:
        panel.set_params({"filter_cutoff_hz": 20_000.0})
    panel.paramsChanged.disconnect(reseed)
    assert panel.filter_cutoff_hz.value() == pytest.approx(9_000.0)
    panel.set_recording(SLOW)
    panel.set_recording(FAST)
    assert panel.filter_cutoff_hz.value() == pytest.approx(9_000.0)


def test_a_re_range_from_inside_a_clamp_does_not_end_it_early(panel, qtbot):
    """A listener can re-range while a clamp is under way. If the inner re-range cleared
    the flag that tells a clamp from an edit, instead of restoring it, the outer one's
    later clamps would be remembered as set: 15 kHz would come back as 12499.99 Hz."""
    panel.set_recording(FAST)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    _type(qtbot, panel.hp_cutoff_hz, "15000")
    inner: list[bool] = []

    def re_range(_params):
        if not inner:
            inner.append(True)
            panel.set_recording(FAST)

    panel.paramsChanged.connect(re_range)
    panel.set_recording(SLOW)                               # the low-pass clamps first
    panel.paramsChanged.disconnect(re_range)
    assert inner == [True]
    panel.set_recording(FAST)
    assert (panel.filter_cutoff_hz.value(),
            panel.hp_cutoff_hz.value()) == pytest.approx((20_000.0, 15_000.0))


def test_an_unknown_rate_puts_back_what_the_static_range_holds(panel, qtbot):
    panel.set_recording(FAST)
    _type(qtbot, panel.filter_cutoff_hz, "20000")
    panel.set_recording(SLOW)
    panel.set_recording(None)                               # a legacy TDMS
    assert panel.filter_cutoff_hz.value() == pytest.approx(20_000.0)
    assert panel.clamp_notice() == ""


# -- the same, through the window -----------------------------------------------------------------

def _doc(tmp_path, name: str, fs: float) -> TraceDocument:
    rng = np.random.default_rng(0)
    n = int(fs)                                             # one second
    sig = (np.full(n, 100.0) + rng.normal(0.0, 1.5, n)).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=n,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture
def window(qapp, qtbot):
    from nanoscope.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _install(window, doc) -> str:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(window._session.document(doc_id))
    return doc_id


def test_switching_recordings_in_the_window_brings_the_typed_value_back(
        window, qtbot, tmp_path):
    fast = _install(window, _doc(tmp_path, "fast.dat", FAST))
    slow = _install(window, _doc(tmp_path, "slow.dat", SLOW))
    det = window._analysis_panel.detection_panel

    window._on_selection(fast, 0)
    _type(qtbot, det.filter_cutoff_hz, "20000")
    window._on_selection(slow, 0)
    assert det.filter_cutoff_hz.value() < SLOW / 2
    window._on_selection(fast, 0)
    assert det.filter_cutoff_hz.value() == pytest.approx(20_000.0)


def test_taking_back_a_lower_sample_rate_declaration_brings_it_back(window, qtbot, tmp_path):
    did = _install(window, _doc(tmp_path, "fast.dat", FAST))
    det = window._analysis_panel.detection_panel
    window._on_selection(did, 0)
    _type(qtbot, det.filter_cutoff_hz, "20000")

    window._session.set_sample_rate_declaration(
        did, SampleRateDeclaration("sample_rate", sample_rate_hz=SLOW))
    assert det.filter_cutoff_hz.value() < SLOW / 2
    window._session.set_sample_rate_declaration(did, None)  # the file's own rate again
    assert det.filter_cutoff_hz.value() == pytest.approx(20_000.0)


# ================================================================================================
# the ML notch box never makes the form a non-pipeline
# ================================================================================================

@pytest.mark.parametrize("text, kept", [
    ("60, inf", (60.0,)),
    ("inf", (60.0, 120.0, 180.0)),                  # nothing usable: the trained lines
    ("1e999", (60.0, 120.0, 180.0)),
    ("+Infinity; 120", (120.0,)),
    ("50 / nan / -inf / abc / 0 / -5 / 100", (50.0, 100.0)),
])
def test_an_unusable_notch_entry_is_dropped_whatever_it_parses_to(panel, text, kept):
    """Infinity parses as a float and is positive, and `MLPreprocessConfig` refuses it: it
    was the one entry that made ``ml_params()`` raise, where every other typo is dropped."""
    panel.ml_notch_freqs.setText(text)
    assert tuple(panel.ml_params()["ml_preprocess"]["notch_freqs"]) == kept


def test_a_switch_of_recording_finishes_with_an_infinite_notch_entry_in_the_ml_box(
        window, qtbot, tmp_path):
    """Measured: ``60, inf`` typed into the ML notch box, then a switch of recording. The seed
    of the arriving recording read the form, ``ml_params()`` raised from inside the writing
    guard, and MainWindow said Analysis could not follow: the arrival was never announced."""
    first = _install(window, _doc(tmp_path, "a.dat", FAST))
    second = _install(window, _doc(tmp_path, "b.dat", FAST))
    panel = window._analysis_panel
    det = panel.detection_panel
    window._on_selection(first, 0)
    det.ml_notch_freqs.setText("60, inf")
    det.ml_notch_freqs.editingFinished.emit()
    arrived: list[tuple] = []
    panel.formArrived.connect(lambda old, new: arrived.append((old, new)))

    window._on_selection(second, 0)
    assert "could not follow" not in window.statusBar().currentMessage(), (
        window.statusBar().currentMessage())
    assert arrived == [((first, 0, None), (second, 0, None))]
    assert panel.form_state()["ml"]["ml_preprocess"]["notch_freqs"] == [60.0]


# ================================================================================================
# deleting a protocol, or a saved ML preset, asks first
# ================================================================================================

def _saved_protocol(panel, monkeypatch, name: str = "mine") -> None:
    _named(monkeypatch, name)
    panel._save_protocol()
    assert panel.protocol_combo.currentText() == name


def test_deleting_a_protocol_asks_and_an_unanswered_prompt_keeps_it(
        panel, monkeypatch, blocked_modals):
    """The suite answers every live modal with no click -- Esc, in effect -- and Esc is
    Cancel here, so the protocol must survive it."""
    _saved_protocol(panel, monkeypatch)
    panel._delete_protocol()
    assert "mine" in DetectionPanel._load_protocols()
    assert panel.protocol_combo.currentText() == "mine"
    assert [m.kind for m in blocked_modals] == ["QMessageBox"]
    assert "“mine”" in blocked_modals[0].text


def test_cancel_keeps_the_protocol_and_delete_forgets_it(panel, monkeypatch):
    _saved_protocol(panel, monkeypatch)
    asked: list[str] = []

    monkeypatch.setattr(DetectionPanel, "_confirm_delete_protocol",
                        lambda self, name: asked.append(name) or False)
    panel._delete_protocol()
    assert asked == ["mine"]
    assert "mine" in DetectionPanel._load_protocols()

    monkeypatch.setattr(DetectionPanel, "_confirm_delete_protocol",
                        lambda self, name: asked.append(name) or True)
    panel._delete_protocol()
    assert asked == ["mine", "mine"]
    assert "mine" not in DetectionPanel._load_protocols()
    assert panel.protocol_combo.findText("mine") < 0


def test_the_delete_prompt_is_plain_text_with_named_buttons_and_cancel_by_default(
        panel, monkeypatch):
    """docs/ARCHITECTURE.md: "Buttons are named, never indexed" -- Return and Esc must
    both keep the protocol. And a name that looks like markup is shown as typed."""
    name = "<b>bold</b> chain"
    _saved_protocol(panel, monkeypatch, name)
    seen: dict = {}

    def exec_(box):
        seen.update(text=box.text(), format=box.textFormat(),
                    default=box.defaultButton().text(), escape=box.escapeButton().text(),
                    buttons=sorted(b.text() for b in box.buttons()))
        next(b for b in box.buttons() if b.text() == "Delete").click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    panel._delete_protocol()
    assert seen["format"] == Qt.PlainText
    assert f"“{name}”" in seen["text"]
    assert seen["buttons"] == ["Cancel", "Delete"]
    assert seen["default"] == "Cancel" and seen["escape"] == "Cancel"
    assert name not in DetectionPanel._load_protocols()     # Delete was clicked


def test_deleting_a_saved_ml_preset_asks_too(panel, monkeypatch, blocked_modals):
    _named(monkeypatch, "mine")
    panel._save_ml_preset_as()
    assert panel.ml_preset_name() == "mine"

    panel._delete_ml_preset()                                # unanswered == Cancel
    assert "mine" in DetectionPanel._load_ml_user_presets()
    assert panel.ml_preset_name() == "mine"
    assert [m.kind for m in blocked_modals] == ["QMessageBox"]
    # It says what a click does not show: the tab falls back to the default preset.
    assert "“mine”" in blocked_modals[0].text and "default preset" in blocked_modals[0].text

    monkeypatch.setattr(DetectionPanel, "_confirm_delete_ml_preset", lambda self, name: True)
    panel._delete_ml_preset()
    assert DetectionPanel._load_ml_user_presets() == {}
    assert panel.ml_preset_name() == DEFAULT_PRESET


# ================================================================================================
# the signals the undo wiring names its steps by
# ================================================================================================

def test_protocol_recalled_fires_once_per_recall_after_the_form_settles(panel):
    events: list[tuple[str, object]] = []
    panel.paramsChanged.connect(lambda p: events.append(("params", None)))
    panel.protocolRecalled.connect(
        lambda name: events.append(("recalled", (name, dict(panel.params())))))

    panel.protocol_combo.setCurrentText(HOST_GUEST)
    recalls = [e for e in events if e[0] == "recalled"]
    assert len(recalls) == 1 and recalls[0][1][0] == HOST_GUEST
    assert events[-1][0] == "recalled"                      # after the last form change
    recipe = panel._protocol_recipe(HOST_GUEST)
    assert recipes_equal(recalls[0][1][1], complete_recipe(recipe))

    panel.protocol_combo.setCurrentText(panel._PROTO_MANUAL)   # not a recall
    panel.protocol_combo.setCurrentText(HOST_GUEST)
    assert [e[1][0] for e in events if e[0] == "recalled"] == [HOST_GUEST, HOST_GUEST]


def test_a_recall_the_form_cannot_hold_exactly_is_still_one_recall(panel, monkeypatch):
    """On a recording too slow for the protocol's low-pass the form clamps it, the combo
    drops to manual and protocolNotice says why -- but the form DID change, once."""
    _saved_protocol(panel, monkeypatch, "wide")
    panel.protocol_combo.setCurrentText(panel._PROTO_MANUAL)
    stored = DetectionPanel._load_protocols()
    stored["wide"]["filter_cutoff_hz"] = 20_000.0
    DetectionPanel._store_protocols(stored)
    panel.set_recording(SLOW)
    recalled: list[str] = []
    notices: list[str] = []
    panel.protocolRecalled.connect(recalled.append)
    panel.protocolNotice.connect(notices.append)

    panel.protocol_combo.setCurrentText("wide")
    assert recalled == ["wide"]
    assert len(notices) == 1 and "could not be applied exactly" in notices[0]
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL


def test_neither_an_unreadable_protocol_nor_a_save_is_a_recall(panel, monkeypatch):
    recalled: list[str] = []
    panel.protocolRecalled.connect(recalled.append)
    _saved_protocol(panel, monkeypatch, "kept")             # Save-as selects, never recalls
    assert recalled == []

    DetectionPanel._store_protocols({"kept": {"notch_q": "not a number"}})
    panel._load_protocols_into_combo()
    notices: list[str] = []
    panel.protocolNotice.connect(notices.append)
    panel.protocol_combo.setCurrentText("kept")
    assert len(notices) == 1 and "the form was not changed" in notices[0]
    assert recalled == []


def test_ml_params_changed_fires_on_an_ml_edit(panel, qtbot):
    sent: list[dict] = []
    panel.mlParamsChanged.connect(sent.append)
    _type(qtbot, panel.ml_merge_ms, "0.7")
    assert len(sent) == 1
    assert sent[0]["ml_decode"]["merge_radius_ms"] == pytest.approx(0.7)
    assert sent[0] == panel.ml_params()

    panel.ml_range.setCurrentIndex(1)                       # also in ml_params()
    assert len(sent) == 2 and sent[-1]["range_source"] == "whole"


def test_ml_params_changed_fires_once_on_set_ml_params_and_it_is_no_preset_load(panel):
    """A seed applies its preset first and lays the stored run over it. That preset is
    a step on the way: announcing it would hand a listener a form the seed never meant
    to leave on screen."""
    sent: list[dict] = []
    loaded: list[str] = []
    panel.mlParamsChanged.connect(sent.append)
    panel.mlPresetLoaded.connect(loaded.append)
    stored = dict(panel.ml_params())
    stored["ml_preset"] = "notebook3"
    stored["ml_decode"] = {**stored["ml_decode"], "preset": "notebook3",
                           "enter_threshold": 0.9, "exit_threshold": 0.3}
    stored["ml_preprocess"] = {**stored["ml_preprocess"], "preset": "notebook3"}

    panel.set_ml_params(stored)
    assert loaded == []
    assert len(sent) == 1
    assert sent[0]["ml_preset"] == "notebook3"
    assert sent[0]["ml_decode"]["enter_threshold"] == pytest.approx(0.9)
    assert sent[0] == panel.ml_params()

    panel.set_ml_params({})                                 # changes nothing, says nothing
    assert len(sent) == 1


def test_a_preset_load_is_announced_once_by_name(panel):
    sent: list[dict] = []
    loaded: list[str] = []
    panel.mlParamsChanged.connect(sent.append)
    panel.mlPresetLoaded.connect(loaded.append)

    panel.ml_preset.setCurrentIndex(panel.ml_preset.findData("notebook1"))   # a pick
    assert loaded == ["notebook1"]
    assert len(sent) == 1 and sent[0]["ml_preset"] == "notebook1"

    panel.apply_ml_preset("notebook2", user_choice=True)
    assert loaded == ["notebook1", "notebook2"]
    assert len(sent) == 2 and sent[-1]["ml_preset"] == "notebook2"
