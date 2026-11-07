"""Shared recipe routes over the established template stores; never run an analysis."""
from dataclasses import dataclass, replace
from copy import deepcopy

from PySide6 import QtWidgets

from .analysis_catalog import ANALYSIS_LABELS, ENTRY_BY_ROUTE
from .plot_appearance import PlotAppearance
from .plot_templates import load_window_templates, store_window_templates, decode_window_template, encode_window_template

STORES = {'distribution_summaries': 'distribution-summaries', 'comparison_figures': 'comparison-figures',
          'event_waveforms': 'event-waveforms', 'noise': 'noise-spectrum', 'voltage_series': 'voltage-dependence'}
SUPPORTED = frozenset(('distributions', *STORES))
UNSUPPORTED = {
    'iv': 'I-V has no separate analytical recipe. Save its reusable look with Style, or Duplicate to retain its figure and current selections.',
    'subpopulations': 'Subpopulations is a generated analysis report, not a plotting recipe. Generate a new report from the analysis controls, or Duplicate to retain the current report and its look.',
}


@dataclass(frozen=True)
class RecipeEntry:
    kind: str
    name: str
    payload: dict
    builtin: bool = False

    @property
    def label(self):
        return f'{ANALYSIS_LABELS[self.kind]} · {self.name}' + (' (built in)' if self.builtin else '')


def entries():
    from .plots_panel import PlotsWindow, plot_config_to_json
    values = [RecipeEntry('distributions', name, deepcopy(payload))
              for name, payload in PlotsWindow._load_templates().items()]
    values += [RecipeEntry('distributions', name, plot_config_to_json(config), True)
               for name, config in PlotsWindow._BUILTIN_TEMPLATES.items()]
    for kind, store in STORES.items():
        values += [RecipeEntry(kind, name, deepcopy(payload)) for name, payload in load_window_templates(store).items()]
    return tuple(sorted(values, key=lambda entry: (entry.kind, entry.name.casefold(), entry.builtin)))


def _store(kind, values):
    if kind == 'distributions':
        from .plots_panel import PlotsWindow
        return PlotsWindow._store_templates(values)
    return store_window_templates(STORES[kind], values)


def _stored(kind):
    if kind == 'distributions':
        from .plots_panel import PlotsWindow
        return PlotsWindow._load_templates()
    return load_window_templates(STORES[kind])


def save_page(page, kind, name, *, overwrite=False):
    name = str(name).strip()
    if not name or len(name) > 100:
        raise ValueError('A recipe name must contain 1–100 characters.')
    if kind not in SUPPORTED:
        raise ValueError(UNSUPPORTED[kind])
    values = _stored(kind)
    if name in values and not overwrite:
        raise ValueError('A recipe with this name already exists.')
    if kind == 'distributions':
        from .plots_panel import plot_config_to_json
        payload = plot_config_to_json(page.config().as_template())
    else:
        settings, appearance = page.template_settings()
        payload = encode_window_template(STORES[kind], settings, appearance)
    values[name] = payload
    if _store(kind, values) is False:
        raise ValueError('The recipe could not be saved.')
    return RecipeEntry(kind, name, deepcopy(payload))


def rename(entry, name):
    name = str(name).strip()
    if not name or len(name) > 100:
        raise ValueError('A recipe name must contain 1–100 characters.')
    values = _stored(entry.kind)
    if entry.builtin or values.get(entry.name) != entry.payload:
        raise ValueError('This recipe changed or is built in; select a saved recipe again.')
    if name != entry.name and name in values:
        raise ValueError('A recipe with this name already exists.')
    values[name] = values.pop(entry.name)
    if _store(entry.kind, values) is False:
        raise ValueError('The recipe could not be renamed.')


def delete(entry):
    values = _stored(entry.kind)
    if entry.builtin or values.get(entry.name) != entry.payload:
        raise ValueError('This recipe changed or is built in; select a saved recipe again.')
    del values[entry.name]
    if _store(entry.kind, values) is False:
        raise ValueError('The recipe could not be deleted.')


def initial_recipe(entry, *, include_words=False):
    """Validate a detached recipe and explicitly choose whether its words travel."""
    if entry.kind == 'distributions':
        from .plots_panel import plot_config_from_json, plot_config_to_json
        template = plot_config_to_json(plot_config_from_json(entry.payload).as_template())
        style = deepcopy(template['style'])
        style.pop('series_styles', None)  # per-series identities belong to the original figure
        if not include_words:
            for key in ('title', 'x_label', 'y_label'):
                style[key] = ''
            style['text_styles'] = {key: {field: value for field, value in values.items() if field != 'text'}
                                    for key, values in style.get('text_styles', {}).items()}
            if style.get('fit_label'):
                style['fit_label']['template'] = ''
        return {'plot_template': template, 'recipe_style': style}
    settings, appearance = decode_window_template(STORES[entry.kind], entry.payload)
    original = PlotAppearance.from_mapping(appearance).template_form()
    look = original.style_only()
    if include_words:
        # Words are independent of look; source legend entries never travel.
        text = dict(look.text or ())
        for key, style in original.text or ():
            if style.text is not None:
                text[key] = replace(text.get(key, style), text=style.text)
        label = look.fit_label
        if original.fit_label is not None:
            label = replace(label or original.fit_label, template=original.fit_label.template)
        look = replace(look, title=original.title, x_label=original.x_label, y_label=original.y_label,
                       text=tuple(sorted(text.items())) or None, fit_label=label)
    return {'window_template': settings, 'recipe_appearance': look.to_dict()}


def seed_appearance(record, recipe):
    """Canonical appearance is ready even when the new figure cannot hydrate yet."""
    from ..core.figure_codecs import encode_legacy, merge_native_capture
    native = encode_legacy(record.spec, record.resources, materialize_defaults=True)
    if record.spec.kind == 'distributions' and 'recipe_style' in recipe:
        from .figure_adapter_plots import _native_config, _with_native_config
        from .plots_panel import StyleSettings, _decode_dc, _project_style_settings
        config = _native_config(native)
        style = _project_style_settings(_decode_dc(StyleSettings, recipe['recipe_style']))
        native = _with_native_config(native, replace(config, style=style))
    elif 'recipe_appearance' in recipe:
        from .figure_adapter_pages import _with_native_appearance
        native = _with_native_appearance(native, record.spec.kind,
                                         PlotAppearance.from_mapping(recipe['recipe_appearance']))
    else:
        return record
    return merge_native_capture(native, record)


def pending_recipe(recipe):
    """Retain only the deferred analysis initializer after appearance is canonical."""
    value = deepcopy(recipe or {})
    value.pop('recipe_style', None)
    value.pop('recipe_appearance', None)
    if isinstance(value.get('plot_template'), dict):
        # The initializer always takes style from the canonical record. Keeping its
        # old recipe copy here would obscure that single authority after a dormant edit.
        value['plot_template'].pop('style', None)
    return value


class RecipeDialog(QtWidgets.QDialog):
    """Choose or manage the same saved recipes from New and Export."""

    def __init__(self, parent=None, *, choose=False, kind=None):
        super().__init__(parent)
        self.setWindowTitle('New figure from recipe' if choose else 'Saved recipes')
        self.resize(620, 460)
        self._choose = choose
        column = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._selection_changed)
        column.addWidget(self.list, 1)
        self.detail = QtWidgets.QLabel(); self.detail.setWordWrap(True)
        column.addWidget(self.detail)
        self.words = QtWidgets.QCheckBox('Include typed title, axis labels, fit wording and captions')
        self.words.setChecked(False)
        self.words.setVisible(choose)
        self.words.setToolTip('Unchecked uses automatic words. The recipe look is always included; sources and computed results are never copied.')
        column.addWidget(self.words)
        if kind in UNSUPPORTED:
            note = QtWidgets.QLabel(UNSUPPORTED[kind]); note.setWordWrap(True); column.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.rename_button = QtWidgets.QPushButton('Rename…'); self.rename_button.clicked.connect(self._rename)
        self.delete_button = QtWidgets.QPushButton('Delete…'); self.delete_button.clicked.connect(self._delete)
        row.addWidget(self.rename_button); row.addWidget(self.delete_button)
        row.addStretch(1)
        self.accept_button = QtWidgets.QPushButton('Create figure' if choose else 'Close')
        self.accept_button.clicked.connect(self.accept)
        row.addWidget(self.accept_button)
        if choose:
            cancel = QtWidgets.QPushButton('Cancel'); cancel.clicked.connect(self.reject); row.addWidget(cancel)
        column.addLayout(row)
        self.refill()

    def refill(self):
        self._entries = entries()
        self.list.clear()
        for entry in self._entries:
            self.list.addItem(entry.label)
        if self._entries:
            self.list.setCurrentRow(0)
        self._selection_changed()

    def selected(self):
        index = self.list.currentRow()
        return self._entries[index] if 0 <= index < len(self._entries) else None

    def _selection_changed(self, *_args):
        entry = self.selected()
        self.rename_button.setEnabled(entry is not None and not entry.builtin)
        self.delete_button.setEnabled(entry is not None and not entry.builtin)
        self.accept_button.setEnabled(entry is not None or not self._choose)
        self.detail.setText((ENTRY_BY_ROUTE[entry.kind].inputs + '\nUses the saved look; words are optional. Update plot runs the analysis.')
                            if entry else 'Save a recipe from a supported figure with Export ▸ Save as recipe…')

    def _rename(self):
        entry = self.selected()
        if entry is None:
            return
        name, accepted = QtWidgets.QInputDialog.getText(self, 'Rename recipe', 'Recipe name:', text=entry.name)
        if accepted:
            try:
                rename(entry, name); self.refill()
            except ValueError as exc:
                self.detail.setText(str(exc))

    def _delete(self):
        entry = self.selected()
        if entry is None:
            return
        if QtWidgets.QMessageBox.question(self, 'Delete recipe?', f'Delete “{entry.name}”? Existing figures are unchanged.') == QtWidgets.QMessageBox.Yes:
            try:
                delete(entry); self.refill()
            except ValueError as exc:
                self.detail.setText(str(exc))
