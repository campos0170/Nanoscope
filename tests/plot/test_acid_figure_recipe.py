import json
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from nanoscope.plot.acid_figure_recipe import capture_figure, reconstruct


def test_roundtrip_preserves_scientific_coordinates_and_editable_panels(tmp_path):
    fig=Figure(figsize=(10,7));FigureCanvasAgg(fig)
    axes=fig.subplots(2,2); x=np.linspace(.1,2,20)
    axes[0,0].plot(x,x*x,label='Trace',color='#127c91');axes[0,0].legend()
    axes[0,0].fill_between(x,x*x-.1,x*x+.1,alpha=.2)
    axes[0,0].set(xlabel='Time (s)',ylabel='Current (pA)',title='Waveform')
    axes[0,1].scatter(x,x**3,c=x,s=np.arange(20)+10,cmap='viridis',label='Events')
    axes[1,0].bar([1,2,3],[10,20,12],color=['red','green','blue']);axes[1,0].axvspan(1,1.5,alpha=.2)
    mesh=axes[1,1].pcolormesh(np.arange(9).reshape(3,3));fig.colorbar(mesh,ax=axes[1,1])
    fig.suptitle('Acid test results')
    path=tmp_path/'plot.png';source=pd.DataFrame({'x':x,'y':x*x})
    capture_figure(fig,path,source)
    rebuilt,data,resources=reconstruct(path)
    assert len(rebuilt.axes)==5
    np.testing.assert_array_equal(rebuilt.axes[0].lines[0].get_xdata(),x)
    np.testing.assert_array_equal(rebuilt.axes[0].lines[0].get_ydata(),x*x)
    np.testing.assert_array_equal(rebuilt.axes[1].collections[0].get_offsets(),np.c_[x,x**3])
    assert rebuilt.axes[0].get_xlabel()=='Time (s)'
    assert rebuilt.axes[0].get_title()=='Waveform'
    assert len(rebuilt.axes[2].patches)==4
    assert len(data)==20
    assert all(not k.endswith('.pkl') for k in resources)
    rebuilt.canvas.draw()
    assert not json.loads(path.with_suffix('.recipe.json').read_text())['warnings']


def test_recipe_rejects_path_traversal(tmp_path):
    path=tmp_path/'bad.recipe.json'
    path.write_text(json.dumps({'format':'nanoscope-acid-figure-recipe','version':1,'arrays':'../outside.npz'}))
    with pytest.raises(ValueError,match='sidecar path'):reconstruct(path)


def test_kde_contours_holes_and_reference_line_transforms_survive(tmp_path):
    import seaborn as sns
    rng=np.random.default_rng(3);x=rng.normal(size=200);y=.6*x+rng.normal(size=200)
    fig=Figure(figsize=(10,4));FigureCanvasAgg(fig);axes=fig.subplots(1,2)
    sns.kdeplot(x=x,y=y,fill=True,levels=5,ax=axes[0])
    sns.kdeplot(x=x,y=y,fill=False,levels=5,ax=axes[1])
    axes[0].axvline(.7,color='red',label='Reference');axes[0].axhline(-.8,color='blue')
    axes[0].legend();fig.canvas.draw()
    path=tmp_path/'kde.png';capture_figure(fig,path)
    restored,_,_=reconstruct(path);restored.canvas.draw()
    assert len(restored.axes[0].collections)==len(axes[0].collections)
    assert sum(len(c.get_paths()) for c in restored.axes[0].collections)>0
    assert restored.axes[0].lines[0].get_transform()==restored.axes[0].get_xaxis_transform()
    assert restored.axes[0].lines[1].get_transform()==restored.axes[0].get_yaxis_transform()
    before=axes[0].collections[0].get_paths()[0]
    after=restored.axes[0].collections[0].get_paths()[0]
    np.testing.assert_array_equal(after.codes,before.codes)
    np.testing.assert_allclose(after.vertices,before.vertices,atol=1e-12)
    assert restored.axes[0].get_legend().get_texts()[0].get_text()=='Reference'


def test_multiple_legends_and_event_identity_survive_and_expire(tmp_path):
    from matplotlib.lines import Line2D
    from nanoscope.plot.study_figure_style import StudyFigureStyle
    fig=Figure();FigureCanvasAgg(fig);axis=fig.subplots()
    axis.plot([0,1,2],[0,3,0],label='Events',color='red')
    first=axis.legend(loc='upper right');axis.add_artist(first)
    axis.legend(handles=[Line2D([],[],color='blue',label='Mean')],loc='lower right')
    axis.set_title('C000123 | block 2');axis.set_xlabel('ms from onset')
    path=tmp_path/'acid_30h_whole_subpopulation'/'event_gallery.png';capture_figure(fig,path)
    rebuilt,_,_=reconstruct(path)
    assert len(rebuilt.axes[0].artists)==1
    assert rebuilt.axes[0].artists[0].get_texts()[0].get_text()=='Events'
    assert rebuilt.axes[0].get_legend().get_texts()[0].get_text()=='Mean'
    identity=rebuilt.axes[0]._nanoscope_event_panel_identity
    style={'event_titles':{identity:'My curated waveform'}}
    StudyFigureStyle(rebuilt).apply(**style)
    assert rebuilt.axes[0].get_title()=='My curated waveform'
    axis.lines[0].set_ydata([0,4,0]);capture_figure(fig,path)
    changed,_,_=reconstruct(path);StudyFigureStyle(changed).apply(**style)
    assert changed.axes[0]._nanoscope_event_panel_identity!=identity
    assert changed.axes[0].get_title()=='C000123 | block 2'


def test_styling_keys_survive_rerender_and_never_cross_recordings(tmp_path):
    from nanoscope.analysis import acid_incubation as engine
    from nanoscope.analysis import manual_workspace as checkpoint
    paths=[]
    for tag in ('acid_2h','acid_10h'):
        figure=Figure();FigureCanvasAgg(figure);axis=figure.subplots()
        axis.plot([0,1],[0,2]);axis.set_title(tag)
        path=tmp_path/tag/(tag+'_subpopulation')/'same_name.png';path.parent.mkdir(parents=True)
        figure.savefig(path);capture_figure(figure,path);paths.append(path)
    state={'schema':engine.SCHEMA,'records':[{'tag':'acid_2h'},{'tag':'acid_10h'},{'tag':'supplement','comparison_only':True}],
           'settings':engine.default_settings(),'history':[],'analysis':{'cohort':[],'selected_event_fingerprints':{}},
           'figures':paths,'figure_styles':{'acid_2h::same_name':{'panels':{'0':{'title':'User title'}}}}}
    # No cohort means this isolated rendering test has no scientific rows to revalidate.
    rendered=engine.render_figures(state,tmp_path/'rendered')
    checkpoint.save_workspace(rendered,tmp_path/'saved')
    reopened=checkpoint.load_workspace(tmp_path/'saved')
    rerendered=engine.render_figures(reopened,tmp_path/'rerendered')
    assert [engine.figure_style_key(rerendered,p) for p in rerendered['figures']]==['acid_2h::same_name','acid_10h::same_name']
    figures=[engine.reconstruct_figure(rerendered,p)[0] for p in rerendered['figures']]
    assert [f.axes[0].get_title() for f in figures]==['User title','acid_10h']
    assert all(p.name=='same_name.png' for p in rerendered['figures'])
    # A new scientific run with the same key must use its new measurements,
    # while inheriting only its appearance choices from the earlier run.
    new_path=tmp_path/'fresh'/'acid_2h_subpopulation'/'same_name.png'
    new_path.parent.mkdir(parents=True)
    changed=Figure();FigureCanvasAgg(changed);changed.subplots().plot([0,1],[0,7])
    changed.savefig(new_path);capture_figure(changed,new_path)
    rerendered['figures']=[new_path]
    fresh=engine.reconstruct_figure(rerendered,new_path)[0]
    np.testing.assert_array_equal(fresh.axes[0].lines[0].get_ydata(),[0,7])
    assert fresh.axes[0].get_title()=='User title'


def test_unfilled_scatter_and_linecollection_legend_handles_roundtrip(tmp_path):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    fig=Figure();FigureCanvasAgg(fig);axis=fig.subplots()
    line=LineCollection([[[0,0],[1,2]]],colors=['black'],label='Envelope');axis.add_collection(line)
    axis.scatter([0,1],[1,2],marker='^',facecolors='none',edgecolors='red',label='Resistive')
    axis.legend();path=tmp_path/'legend.png';capture_figure(fig,path)
    recreated,_,_=reconstruct(path)
    handles=recreated.axes[0].get_legend().legend_handles
    assert len(handles)==2 and all(isinstance(h,Line2D) for h in handles)
    assert handles[0].get_color()==[0.,0.,0.,1.]
    assert handles[1].get_markerfacecolor()=='none'
    assert handles[1].get_markeredgecolor()==[1.,0.,0.,1.]
    np.testing.assert_array_equal(handles[1].get_marker().vertices,axis.collections[1].get_paths()[0].vertices)
    recreated.canvas.draw()


def test_symlog_threshold_preserves_calibration_display_coordinates(tmp_path):
    fig=Figure();FigureCanvasAgg(fig);axis=fig.subplots()
    axis.set_xscale('symlog',linthresh=.001,linscale=1.,base=10)
    axis.plot([0,.001,.01,.1],[.5,.6,.8,.9]);axis.axvline(.01)
    axis.set_ylim(.4,1.);fig.canvas.draw()
    path=tmp_path/'calibration.png';capture_figure(fig,path)
    restored,_,_=reconstruct(path);restored.canvas.draw()
    x=np.array([[0,.5],[.0001,.5],[.001,.6],[.01,.8],[.1,.9]])
    np.testing.assert_allclose(restored.axes[0].transData.transform(x),axis.transData.transform(x),atol=1e-9)
    assert restored.axes[0].xaxis.get_transform().linthresh==.001


def test_nondefault_log_base_minor_ticks_and_mask_policy_survive(tmp_path):
    fig=Figure();FigureCanvasAgg(fig);axis=fig.subplots()
    axis.set_xscale('log',base=2,subs=[1,1.5],nonpositive='mask')
    axis.set_yscale('symlog',base=3,linthresh=.02,linscale=2.,subs=[1,2])
    axis.plot([.25,.5,1,2,4],[-1,-.02,0,.02,1]);fig.canvas.draw()
    path=tmp_path/'nondefault.png';capture_figure(fig,path)
    restored,_,_=reconstruct(path);restored.canvas.draw()
    points=np.array([[.25,-1],[.5,-.02],[1,0],[2,.02],[4,1]])
    np.testing.assert_allclose(restored.axes[0].transData.transform(points),axis.transData.transform(points),atol=1e-9)
    np.testing.assert_array_equal(restored.axes[0].xaxis.get_minorticklocs(),axis.xaxis.get_minorticklocs())
    np.testing.assert_array_equal(restored.axes[0].yaxis.get_minorticklocs(),axis.yaxis.get_minorticklocs())
    assert np.isnan(restored.axes[0].xaxis.get_transform().transform([-1]))[0]


def test_joint_kde_unfilled_contours_and_hidden_marginal_labels_survive(tmp_path):
    from nanoscope.analysis.acid_incubation import _AcidPlot
    import seaborn as sns
    import pandas as pd
    plot=_AcidPlot();grid=plot.joint_grid(height=5,ratio=4)
    rng=np.random.default_rng(62)
    x=pd.Series(rng.normal(size=160),name='dwell_ms')
    y=pd.Series(.6*x+rng.normal(size=160),name='peak_fracchange')
    sns.kdeplot(x=x,y=y,ax=grid.ax_joint,levels=6,color='brown')
    sns.kdeplot(x=x,ax=grid.ax_marg_x,fill=True,alpha=.15,color='brown')
    sns.kdeplot(y=y,ax=grid.ax_marg_y,fill=True,alpha=.15,color='brown')
    grid.figure.canvas.draw();path=tmp_path/'joint.png';capture_figure(grid.figure,path)
    restored,_,_=reconstruct(path);restored.canvas.draw()
    for before,after in zip(grid.figure.axes,restored.axes):
        assert before.xaxis.label.get_visible()==after.xaxis.label.get_visible()
        assert before.yaxis.label.get_visible()==after.yaxis.label.get_visible()
        for a,b in zip(before.collections,after.collections):
            np.testing.assert_allclose(a.get_facecolors(),b.get_facecolors(),atol=1e-12)
            np.testing.assert_allclose(a.get_edgecolors(),b.get_edgecolors(),atol=1e-12)
    assert len(restored.axes[0].collections[0].get_facecolors())==0
