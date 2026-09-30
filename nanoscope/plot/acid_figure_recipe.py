"""Portable numerical Matplotlib artist recipes for editable acid-study figures.

Recipes contain JSON metadata and finite-type NPZ arrays, never pickled figures,
Python code, or expressions. All coordinates are display evidence, not refitted
scientific results. Unsupported artists stop recipe creation rather than disappearing on restyling.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, QuadMesh
from matplotlib.colors import Normalize, LogNorm, ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.text import Annotation
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

VERSION = 1


def _plain(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    if isinstance(value, dict): return {str(k):_plain(v) for k,v in value.items()}
    if value is None or isinstance(value, (str,bool,int,float)): return value
    return str(value)


def _color(value):
    try: return list(to_rgba(value))
    except (ValueError,TypeError): return value


def _font(text):
    return dict(fontsize=float(text.get_fontsize()),fontfamily=text.get_fontfamily(),
                fontstyle=text.get_fontstyle(),fontweight=text.get_fontweight(),
                color=_color(text.get_color()))


def capture_figure(fig, path, source=None):
    """Write a safe numeric recipe beside ``path`` without modifying the Figure."""
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    # Constrained/tight layout and colourbar axis positions must be resolved first.
    fig.canvas.draw()
    arrays = {}; warnings=[]
    def put(value):
        value = np.ma.asarray(value)
        if value.dtype.kind not in 'biufc': raise ValueError('Figure coordinate arrays must be numeric.')
        value = np.asarray(value.filled(np.nan) if np.ma.isMaskedArray(value) and np.any(value.mask) else value)
        name=f'a{len(arrays)}'; arrays[name]=value; return name
    def cmap_recipe(cmap):
        return dict(cmap=cmap.name,cmap_lut=put(cmap(np.linspace(0,1,cmap.N))),
                    cmap_bad=list(cmap.get_bad()),cmap_under=list(cmap.get_under()),cmap_over=list(cmap.get_over()))
    def text_recipe(text, axis=None):
        if isinstance(text,Annotation) and text.arrow_patch is not None:
            raise ValueError('This figure contains an annotation arrow that cannot yet be edited safely.')
        trans = text.get_transform()
        coord = 'axes' if axis is not None and trans == axis.transAxes else 'data' if axis is not None and trans == axis.transData else 'figure'
        position = list(text.get_position())
        if coord == 'figure': position = list(fig.transFigure.inverted().transform(trans.transform(position)))
        return dict(text=text.get_text(),position=position,coordinates=coord,rotation=float(text.get_rotation()),
                    ha=text.get_ha(),va=text.get_va(),visible=text.get_visible(),
                    figure_title=text is getattr(fig,'_suptitle',None),**_font(text))
    def scale_parameters(axis):
        scale=axis.get_scale();transform=axis.get_transform()
        if scale=='linear':return {}
        subs=getattr(axis._scale,'subs',None)
        subs=None if subs is None else list(subs)
        if scale=='log':return {'base':float(transform.base),'subs':subs,'nonpositive':'clip' if transform._clip else 'mask'}
        if scale=='symlog':return {'base':float(transform.base),'linthresh':float(transform.linthresh),'linscale':float(transform.linscale),'subs':subs}
        raise ValueError('Unsupported editable axis scale: '+scale)
    def legend_recipe(legend):
        proxies=[]
        for handle in legend.legend_handles:
            if isinstance(handle,Line2D):
                proxies.append(dict(type='line',color=_color(handle.get_color()),linewidth=float(handle.get_linewidth()),
                    linestyle=handle.get_linestyle(),marker=handle.get_marker(),markersize=float(handle.get_markersize()),
                    markerfacecolor=_color(handle.get_markerfacecolor()),markeredgecolor=_color(handle.get_markeredgecolor())))
            elif isinstance(handle,LineCollection):
                colors=handle.get_colors(); widths=handle.get_linewidths()
                proxies.append(dict(type='line',color=colors[0].tolist() if len(colors) else 'none',
                    linewidth=float(widths[0]) if len(widths) else 1.,linestyle='-',marker='None',
                    markersize=6,markerfacecolor='none',markeredgecolor='none'))
            elif isinstance(handle,PathCollection):
                paths=handle.get_paths();sizes=handle.get_sizes();fc=handle.get_facecolors();ec=handle.get_edgecolors()
                if len(paths)!=1:raise ValueError('Unsupported multi-path scatter legend.')
                proxies.append(dict(type='scatter',vertices=put(paths[0].vertices),
                    codes=put(paths[0].codes) if paths[0].codes is not None else None,
                    markersize=float(np.sqrt(sizes[0])) if len(sizes) else 6.,
                    markerfacecolor=fc[0].tolist() if len(fc) else 'none',
                    markeredgecolor=ec[0].tolist() if len(ec) else 'none',
                    markeredgewidth=float(handle.get_linewidths()[0]) if len(handle.get_linewidths()) else 1.,
                    alpha=handle.get_alpha()))
            elif isinstance(handle,Patch):
                proxies.append(dict(type='patch',facecolor=list(handle.get_facecolor()),
                    edgecolor=list(handle.get_edgecolor()),linewidth=float(handle.get_linewidth()),
                    hatch=handle.get_hatch(),alpha=handle.get_alpha()))
            else: raise ValueError('Unsupported editable legend handle: '+type(handle).__name__)
        anchor=legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        return dict(labels=[t.get_text() for t in legend.get_texts()],
            fontsize=float(legend.get_texts()[0].get_fontsize()) if legend.get_texts() else 10,
            frameon=legend.get_frame_on(),loc=legend._loc,ncols=getattr(legend,'_ncols',1),title=legend.get_title().get_text(),
            proxies=proxies,anchor=list(anchor.bounds))
    recipe = dict(format='nanoscope-acid-figure-recipe',version=VERSION,figsize=list(fig.get_size_inches()),
                  dpi=float(fig.dpi),facecolor=list(fig.get_facecolor()),axes=[],texts=[],warnings=warnings)
    for axis in fig.axes:
        if axis.tables or axis.child_axes:
            raise ValueError('This figure contains an unsupported editable table or inset axis.')
        entry=dict(position=list(axis.get_position().bounds),xscale=axis.get_xscale(),yscale=axis.get_yscale(),
            xscale_parameters=scale_parameters(axis.xaxis),yscale_parameters=scale_parameters(axis.yaxis),
            xlim=list(axis.get_xlim()),ylim=list(axis.get_ylim()),xlabel=axis.get_xlabel(),ylabel=axis.get_ylabel(),
            title=axis.get_title(),title_font=_font(axis.title),xlabel_font=_font(axis.xaxis.label),ylabel_font=_font(axis.yaxis.label),
            title_visible=axis.title.get_visible(),xlabel_visible=axis.xaxis.label.get_visible(),ylabel_visible=axis.yaxis.label.get_visible(),
            xticks=put(axis.get_xticks()),yticks=put(axis.get_yticks()),
            xticklabels=[t.get_text() for t in axis.get_xticklabels()],yticklabels=[t.get_text() for t in axis.get_yticklabels()],
            x_tick_rotation=[float(t.get_rotation()) for t in axis.get_xticklabels()],
            y_tick_rotation=[float(t.get_rotation()) for t in axis.get_yticklabels()],
            tick_fontsize=float(axis.get_xticklabels()[0].get_fontsize()) if axis.get_xticklabels() else 10.,
            facecolor=list(axis.get_facecolor()),visible=axis.get_visible(),axis_on=axis.axison,
            grid_x=any(line.get_visible() for line in axis.get_xgridlines()),
            grid_y=any(line.get_visible() for line in axis.get_ygridlines()),
            x_label_position=axis.xaxis.get_label_position(),y_label_position=axis.yaxis.get_label_position(),
            x_tick_position=axis.xaxis.get_ticks_position(),y_tick_position=axis.yaxis.get_ticks_position(),
            spines={k:v.get_visible() for k,v in axis.spines.items()},lines=[],collections=[],patches=[],images=[],texts=[])
        if hasattr(axis,'_nanoscope_event_panel_identity'):
            entry['event_identity']=axis._nanoscope_event_panel_identity
        else:
            event_ids=re.findall(r'\b[CSTRU]\d{6}\b',axis.get_title())
            if event_ids:
                # Captions follow the exact drawn waveform and scientific title,
                # never merely an ordinal panel or reused event number.
                scope=next((p.name for p in path.parents if p.name.endswith('_subpopulation')),'study')
                digest=hashlib.sha256(json.dumps([scope,path.stem,axis.get_title(),event_ids]).encode())
                for line in axis.lines: digest.update(np.asarray(line.get_xydata(),dtype='<f8').tobytes())
                entry['event_identity']=digest.hexdigest()
        for line in axis.lines:
            try:
                x,y=line.get_xdata(orig=False),line.get_ydata(orig=False)
                transform=line.get_transform()
                if transform == axis.transData: coordinates='data'
                elif transform == axis.get_xaxis_transform(): coordinates='xdata_yaxes'
                elif transform == axis.get_yaxis_transform(): coordinates='xaxes_ydata'
                elif transform == axis.transAxes: coordinates='axes'
                else:
                    frozen=axis.transData.inverted().transform(transform.transform(np.c_[x,y]))
                    x,y=frozen[:,0],frozen[:,1];coordinates='data'
                entry['lines'].append(dict(x=put(x),y=put(y),coordinates=coordinates,
                    color=_color(line.get_color()),linewidth=float(line.get_linewidth()),linestyle=line.get_linestyle(),
                    marker=line.get_marker(),markersize=float(line.get_markersize()),
                    markerfacecolor=_color(line.get_markerfacecolor()),markeredgecolor=_color(line.get_markeredgecolor()),
                    markeredgewidth=float(line.get_markeredgewidth()),label=line.get_label(),alpha=line.get_alpha(),
                    zorder=float(line.get_zorder()),drawstyle=line.get_drawstyle(),visible=line.get_visible()))
            except (TypeError,ValueError) as exc: raise ValueError('Figure line cannot be safely made editable: '+str(exc)) from exc
        for collection in axis.collections:
            common=dict(label=collection.get_label(),alpha=collection.get_alpha(),zorder=float(collection.get_zorder()),
                        facecolors=put(collection.get_facecolors()),edgecolors=put(collection.get_edgecolors()),
                        linewidths=put(collection.get_linewidths()),visible=collection.get_visible())
            scalar=collection.get_array()
            if scalar is not None:
                common.update(values=put(scalar),**cmap_recipe(collection.get_cmap()),
                    clim=list(collection.get_clim()),norm='log' if isinstance(collection.norm,LogNorm) else 'linear')
            if isinstance(collection,QuadMesh):
                common.update(type='quadmesh',coordinates=put(collection.get_coordinates()))
            elif isinstance(collection,PathCollection):
                common.update(type='scatter',offsets=put(collection.get_offsets()),sizes=put(collection.get_sizes()),
                    paths=[dict(vertices=put(p.vertices),codes=put(p.codes) if p.codes is not None else None) for p in collection.get_paths()])
            elif isinstance(collection,LineCollection): common.update(type='linecollection',segments=[put(v) for v in collection.get_segments()])
            elif hasattr(collection,'get_paths'):
                # ContourSet / QuadContourSet use compound paths with MOVETO and
                # CLOSEPOLY codes. Keeping codes preserves filled contour holes.
                paths=[]
                for p in collection.get_paths():
                    vertices=axis.transData.inverted().transform(collection.get_transform().transform(p.vertices))
                    paths.append(dict(vertices=put(vertices),codes=put(p.codes) if p.codes is not None else None))
                if not paths: raise ValueError('Unsupported empty collection: '+type(collection).__name__)
                common.update(type='paths',paths=paths)
            else:
                raise ValueError('Figure contains an unsupported editable collection: '+type(collection).__name__)
            entry['collections'].append(common)
        for patch in [*axis.patches,*[v for k,v in axis.spines.items() if k not in ('left','right','top','bottom')]]:
            common=dict(facecolor=list(patch.get_facecolor()),edgecolor=list(patch.get_edgecolor()),
                        linewidth=float(patch.get_linewidth()),linestyle=patch.get_linestyle(),alpha=patch.get_alpha(),
                        hatch=patch.get_hatch(),label=patch.get_label(),zorder=float(patch.get_zorder()),visible=patch.get_visible())
            # Mixed transforms (e.g. axvspan) are frozen into data coordinates.
            vertices=patch.get_path().vertices
            vertices=axis.transData.inverted().transform(patch.get_transform().transform(vertices))
            common.update(type='path',vertices=put(vertices),codes=put(patch.get_path().codes) if patch.get_path().codes is not None else None)
            entry['patches'].append(common)
        for item in axis.images:
            entry['images'].append(dict(values=put(item.get_array()),extent=list(item.get_extent()),origin=item.origin,
                interpolation=item.get_interpolation(),aspect=axis.get_aspect(),**cmap_recipe(item.get_cmap()),
                clim=list(item.get_clim()),norm='log' if isinstance(item.norm,LogNorm) else 'linear',
                alpha=item.get_alpha(),zorder=float(item.get_zorder())))
        for item in axis.texts: entry['texts'].append(text_recipe(item,axis))
        legends=[a for a in axis.artists if isinstance(a,Legend)]
        if axis.get_legend() is not None and axis.get_legend() not in legends: legends.append(axis.get_legend())
        unsupported=[a for a in axis.artists if not isinstance(a,Legend)]
        if unsupported: raise ValueError('Unsupported editable axis artist: '+type(unsupported[0]).__name__)
        entry['legends']=[legend_recipe(legend) for legend in legends]
        recipe['axes'].append(entry)
    for item in fig.texts: recipe['texts'].append(text_recipe(item))
    recipe['legends']=[legend_recipe(legend) for legend in fig.legends]
    if fig.artists: raise ValueError('Unsupported editable figure-level artist: '+type(fig.artists[0]).__name__)
    archive=path.with_suffix('.artists.npz'); metadata=path.with_suffix('.recipe.json')
    np.savez_compressed(archive,**arrays)
    recipe['arrays']=archive.name
    if isinstance(source,pd.DataFrame) and len(source.columns):
        source_path=path.with_suffix('.data.csv'); source.to_csv(source_path,index=False); recipe['source']=source_path.name
    metadata.write_text(json.dumps(recipe,indent=2,default=_plain,allow_nan=False))
    return metadata


def reconstruct(path):
    """Rebuild editable artists from the numeric recipe; never execute saved data."""
    path=Path(path); metadata=path if path.suffix=='.json' else path.with_suffix('.recipe.json')
    recipe=json.loads(metadata.read_text())
    if recipe.get('format')!='nanoscope-acid-figure-recipe' or recipe.get('version')!=VERSION:
        raise ValueError('Unsupported acid figure recipe.')
    if recipe.get('warnings'):
        raise ValueError('This older recipe did not preserve every artist. Regenerate the figure before restyling.')
    def local(name):
        if not isinstance(name,str) or Path(name).name!=name: raise ValueError('Invalid figure sidecar path.')
        return metadata.parent/name
    archive=local(recipe['arrays'])
    with np.load(archive,allow_pickle=False) as loaded: arrays={k:loaded[k] for k in loaded.files}
    def arr(key):
        if key not in arrays: raise ValueError('Missing figure coordinate array.')
        return arrays[key]
    fig=Figure(figsize=recipe['figsize'],dpi=recipe['dpi'],facecolor=recipe['facecolor']); FigureCanvasAgg(fig)
    def add_text(spec,axis=None):
        kw={k:spec[k] for k in ('fontsize','fontfamily','fontstyle','fontweight','color','ha','va','rotation','visible')}
        if spec.get('figure_title'):
            fig.suptitle(spec['text'],x=spec['position'][0],y=spec['position'][1],**kw)
        elif axis is None or spec['coordinates']=='figure': fig.text(*spec['position'],spec['text'],**kw)
        else: axis.text(*spec['position'],spec['text'],transform=axis.transAxes if spec['coordinates']=='axes' else axis.transData,**kw)
    def norm(spec):
        lo,hi=spec['clim']; return LogNorm(lo,hi) if spec.get('norm')=='log' else Normalize(lo,hi)
    def cmap(spec):
        if 'cmap_lut' not in spec:
            from matplotlib import colormaps
            if spec.get('cmap') not in colormaps:
                raise ValueError('This older figure recipe did not preserve its custom color map. Regenerate the figure to style it safely.')
            return colormaps[spec['cmap']]
        result=ListedColormap(arr(spec['cmap_lut']),name=spec.get('cmap','saved'))
        result.set_bad(spec['cmap_bad']);result.set_under(spec['cmap_under']);result.set_over(spec['cmap_over'])
        return result
    def legend_handles(v):
        handles=[]
        for proxy in v.get('proxies',[]):
            kw={k:val for k,val in proxy.items() if k!='type'}
            if proxy['type']=='scatter':
                vertices=arr(kw.pop('vertices'));codes=kw.pop('codes')
                handles.append(Line2D([],[],linestyle='None',marker=MplPath(vertices,arr(codes) if codes else None),**kw))
            elif proxy['type']=='line':handles.append(Line2D([],[],**kw))
            else:
                if kw.get('facecolor')==[] or kw.get('edgecolor')==[]:
                    raise ValueError('This older recipe did not preserve an unfilled legend handle. Regenerate the figure before styling.')
                handles.append(Patch(**kw))
        return handles
    for spec in recipe['axes']:
        axis=fig.add_axes(spec['position'])
        for direction in ('x','y'):
            scale=spec[direction+'scale'];parameters=spec.get(direction+'scale_parameters',{})
            if scale=='symlog' and not parameters:
                raise ValueError('This older recipe omitted its symlog threshold. Regenerate the figure before styling.')
            getattr(axis,'set_'+direction+'scale')(scale,**parameters)
        axis.set_facecolor(spec['facecolor']);axis.set_visible(spec['visible'])
        if not spec['axis_on']: axis.set_axis_off()
        if 'event_identity' in spec: axis._nanoscope_event_panel_identity=spec['event_identity']
        for v in spec['lines']:
            kw={k:v[k] for k in ('color','linewidth','linestyle','marker','markersize','markerfacecolor','markeredgecolor','markeredgewidth','label','alpha','zorder','drawstyle','visible')}
            transforms={'data':axis.transData,'xdata_yaxes':axis.get_xaxis_transform(),
                        'xaxes_ydata':axis.get_yaxis_transform(),'axes':axis.transAxes}
            axis.plot(arr(v['x']),arr(v['y']),transform=transforms.get(v.get('coordinates','data'),axis.transData),**kw)
        for v in spec['collections']:
            common=dict(label=v['label'],alpha=v['alpha'],zorder=v['zorder'],visible=v['visible'],
                        facecolors=arr(v['facecolors']) if len(arr(v['facecolors'])) else 'none',
                        edgecolors=arr(v['edgecolors']) if len(arr(v['edgecolors'])) else 'none',linewidths=arr(v['linewidths']))
            if v['type']=='scatter':
                paths=[MplPath(arr(p['vertices']),arr(p['codes']) if p['codes'] else None) for p in v['paths']]
                item=PathCollection(paths,sizes=arr(v['sizes']),offsets=arr(v['offsets']),offset_transform=axis.transData,**common)
                item.set_transform(Affine2D()); axis.add_collection(item)
            elif v['type']=='polycollection':
                item=PolyCollection([arr(k) for k in v['vertices']],**common);axis.add_collection(item)
            elif v['type']=='linecollection':
                common.pop('facecolors');item=LineCollection([arr(k) for k in v['segments']],**common);axis.add_collection(item)
            elif v['type']=='paths':
                paths=[MplPath(arr(p['vertices']),arr(p['codes']) if p['codes'] else None) for p in v['paths']]
                item=PathCollection(paths,**common);item.set_transform(axis.transData);axis.add_collection(item)
            elif v['type']=='quadmesh':
                coords=arr(v['coordinates']); values=arr(v['values'])
                item=axis.pcolormesh(coords[:,:,0],coords[:,:,1],values.reshape(coords.shape[0]-1,coords.shape[1]-1),
                    cmap=cmap(v),norm=norm(v),shading='flat',alpha=v['alpha'],zorder=v['zorder'])
            else: raise ValueError('Unsupported saved artist type.')
            if 'values' in v:
                item.set_array(arr(v['values']));item.set_cmap(cmap(v));item.set_norm(norm(v))
        for v in spec['patches']:
            kw={k:v[k] for k in ('facecolor','edgecolor','linewidth','linestyle','alpha','hatch','label','zorder','visible')}
            axis.add_patch(PathPatch(MplPath(arr(v['vertices']),arr(v['codes']) if v.get('codes') else None),**kw))
        for v in spec['images']:
            axis.imshow(arr(v['values']),extent=v['extent'],origin=v['origin'],interpolation=v['interpolation'],
                aspect=v['aspect'],cmap=cmap(v),norm=norm(v),alpha=v['alpha'],zorder=v['zorder'])
        axis.set_title(spec['title'],**spec['title_font']);axis.set_xlabel(spec['xlabel'],**spec['xlabel_font']);axis.set_ylabel(spec['ylabel'],**spec['ylabel_font'])
        axis.title.set_visible(spec.get('title_visible',True));axis.xaxis.label.set_visible(spec.get('xlabel_visible',True));axis.yaxis.label.set_visible(spec.get('ylabel_visible',True))
        axis.set_xticks(arr(spec['xticks']),spec['xticklabels']);axis.set_yticks(arr(spec['yticks']),spec['yticklabels'])
        axis.tick_params(labelsize=spec['tick_fontsize'])
        axis.xaxis.set_label_position(spec.get('x_label_position','bottom'));axis.yaxis.set_label_position(spec.get('y_label_position','left'))
        for axis_object,key in ((axis.xaxis,'x_tick_position'),(axis.yaxis,'y_tick_position')):
            position=spec.get(key)
            if position in ('top','bottom','left','right','both','none','default'):axis_object.set_ticks_position(position)
        for tick,rotation in zip(axis.get_xticklabels(),spec['x_tick_rotation']): tick.set_rotation(rotation)
        for tick,rotation in zip(axis.get_yticklabels(),spec['y_tick_rotation']): tick.set_rotation(rotation)
        axis.set_xlim(spec['xlim']);axis.set_ylim(spec['ylim'])
        axis.grid(spec['grid_x'],axis='x');axis.grid(spec['grid_y'],axis='y')
        for key,visible in spec['spines'].items():
            if key in axis.spines: axis.spines[key].set_visible(visible)
        for v in spec['texts']: add_text(v,axis)
        legends=spec.get('legends',[spec['legend']] if 'legend' in spec else [])
        for index,v in enumerate(legends):
            handles=legend_handles(v)
            if handles:
                legend=axis.legend(handles,v['labels'],fontsize=v['fontsize'],frameon=v['frameon'],loc=v['loc'],ncols=v['ncols'],title=v['title'],bbox_to_anchor=v['anchor'],bbox_transform=fig.transFigure)
                if index<len(legends)-1:axis.add_artist(legend)
    for v in recipe['texts']: add_text(v)
    for v in recipe.get('legends',[]):
        handles=legend_handles(v)
        if handles:fig.legend(handles,v['labels'],fontsize=v['fontsize'],frameon=v['frameon'],loc=v['loc'],ncols=v['ncols'],title=v['title'],bbox_to_anchor=v['anchor'],bbox_transform=fig.transFigure)
    try: data=pd.read_csv(local(recipe['source'])) if recipe.get('source') else pd.DataFrame()
    except pd.errors.EmptyDataError: data=pd.DataFrame()
    resources={metadata.name:metadata.read_bytes(),archive.name:archive.read_bytes()}
    return fig,data,resources
