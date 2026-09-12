"""Offline ReportLab fallback for the self-contained manual HTML.

Run with a Python environment containing reportlab, Pillow and pypdf. No browser,
network, Markdown parser or application imports are needed. The HTML is produced
and validated by gen_page.py first. Every prose/table/figure block is retained.
"""
from __future__ import annotations

import argparse
import base64
import html
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
    Spacer, PageBreak, Image, LongTable, TableStyle, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag=tag; self.attrs=dict(attrs); self.children=[]
    def find(self, tag):
        if self.tag==tag: return self
        return next((hit for child in self.children if isinstance(child,Node)
                     and (hit:=child.find(tag)) is not None), None)
    def text(self):
        return ''.join(x.text() if isinstance(x,Node) else x for x in self.children)


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.root=Node(); self.stack=[self.root]
    def handle_starttag(self,tag,attrs):
        node=Node(tag,attrs); self.stack[-1].children.append(node)
        if tag not in {'img','br','hr','meta','input','link'}: self.stack.append(node)
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if self.stack[-1].tag==tag: self.stack.pop()
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:
                del self.stack[i:]; break
    def handle_data(self,data): self.stack[-1].children.append(data)


def clean(text):
    return (text.replace('\u00a0',' ').replace('\u202f',' ').replace('⚠','[!]')
            .replace('⃤',' [!]').replace('⟵','←').replace('⟶','→'))


def escaped_text(text):
    """Retain scientific symbols and UI glyphs, including Unicode subscripts."""
    result=[]
    for char in clean(text):
        escaped=html.escape(char)
        if char.isspace() or pdfmetrics.getFont('Manual').face.charToGlyph.get(ord(char)):
            result.append(escaped)
        elif pdfmetrics.getFont('ManualUnicode').face.charToGlyph.get(ord(char)):
            result.append('<font name="ManualUnicode">'+escaped+'</font>')
        else:
            raise ValueError(f'No PDF font supports {char!r} (U+{ord(char):04X})')
    return ''.join(result)


def inline(node):
    if isinstance(node,str): return escaped_text(node)
    content=''.join(inline(c) for c in node.children)
    if node.tag in {'b','strong'}: return '<b>'+content+'</b>'
    if node.tag in {'i','em'}: return '<i>'+content+'</i>'
    if node.tag=='code': return '<font name="ManualMono" size="8.5">'+content+'</font>'
    if node.tag=='br': return '<br/>'
    if node.tag=='span' and node.attrs.get('class') in {'note','corr','grp','kbd'}:
        return '<br/>'+content
    return content


def fonts():
    base=Path('/System/Library/Fonts/Supplemental')
    names={'Manual':'Arial.ttf','Manual-Bold':'Arial Bold.ttf',
           'Manual-Italic':'Arial Italic.ttf','Manual-BoldItalic':'Arial Bold Italic.ttf',
           'ManualMono':'Courier New.ttf','ManualUnicode':'Arial Unicode.ttf'}
    for name,file in names.items():
        pdfmetrics.registerFont(TTFont(name,str(base/file)))
    pdfmetrics.registerFontFamily('Manual',normal='Manual',bold='Manual-Bold',
        italic='Manual-Italic',boldItalic='Manual-BoldItalic')
    pdfmetrics.registerFontFamily('ManualUnicode',normal='ManualUnicode',bold='ManualUnicode',
        italic='ManualUnicode',boldItalic='ManualUnicode')


def render(source:Path, output:Path):
    fonts()
    tree=Tree(); tree.feed(source.read_text())
    main=tree.root.find('main'); assert main
    width=A4[0]-30*mm
    styles={}
    common=dict(fontName='Manual',fontSize=9.7,leading=14,spaceAfter=8,
                textColor=colors.HexColor('#193044'),splitLongWords=True,
                allowWidows=0,allowOrphans=0)
    styles['body']=ParagraphStyle('body',**common)
    for name,size,leading in [('h1',32,38),('h2',19,24),('h3',13,18),('h4',11,15)]:
        styles[name]=ParagraphStyle(name,parent=styles['body'],fontName='Manual-Bold',
            fontSize=size,leading=leading,spaceBefore=12,spaceAfter=12,keepWithNext=True)
    styles['caption']=ParagraphStyle('caption',parent=styles['body'],fontSize=8.2,
        leading=11,textColor=colors.HexColor('#546574'),spaceAfter=12)
    styles['cell']=ParagraphStyle('cell',parent=styles['body'],fontSize=7.8,leading=10.5,spaceAfter=3)
    styles['small']=ParagraphStyle('small',parent=styles['body'],fontSize=8.3,leading=11.5)
    styles['readout']=ParagraphStyle('readout',parent=styles['small'],fontName='ManualMono',
        fontSize=7.5,leading=10,backColor=colors.HexColor('#f0f4f7'),borderPadding=6,spaceAfter=10)

    class ManualDoc(BaseDocTemplate):
        def __init__(self,path):
            super().__init__(str(path),pagesize=A4,leftMargin=15*mm,rightMargin=15*mm,
                topMargin=16*mm,bottomMargin=17*mm,title='NanoScope User Manual - September 2026',
                author='NanoScope',subject='Specialist workflows and control reference; source 703a96a')
            self.addPageTemplates(PageTemplate(id='manual',frames=[Frame(15*mm,17*mm,width,
                A4[1]-33*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],
                onPage=self.page_footer))
        def page_footer(self,canvas,doc):
            canvas.setFont('Manual',7.5); canvas.setFillColor(colors.HexColor('#61717d'))
            canvas.drawString(15*mm,10*mm,'NanoScope 0.1.0 | Manual revised 2026-09-11 | Source 703a96a')
            canvas.drawRightString(A4[0]-15*mm,10*mm,str(doc.page))
        def afterFlowable(self,flow):
            if hasattr(flow,'_toc_title'):
                key=flow._bookmark; self.canv.bookmarkPage(key)
                level=getattr(flow,'_outline_level',0)
                self.canv.addOutlineEntry(flow._toc_title,key,level=level,closed=False)
                if level==0:
                    self.notify('TOCEntry',(0,flow._toc_title,self.page,key))

    story=[Spacer(1,28*mm),Paragraph('NANOSCOPE',styles['h1']),
        Paragraph('Specialist user manual',styles['h2']),
        Paragraph('Analysis and plotting, step by step.<br/>Recording-specific references. Event characterization. Reproducible figures.',styles['body']),
        Spacer(1,10*mm),Paragraph('Revised 11 September 2026<br/>NanoScope 0.1.0 · Source revision 703a96a',styles['body']),
        Paragraph('Current workflows and interface captures are distinguished from dated historical examples. '
                  'Scientific settings, assumptions and provenance remain explicit.',styles['body']),
        PageBreak(),Paragraph('Contents',styles['h2'])]
    toc=TableOfContents(); toc.levelStyles=[ParagraphStyle('toc',parent=styles['body'],
        fontSize=9,leading=13,spaceAfter=5,rightIndent=18)]
    story.extend([toc,PageBreak()])
    counts={'figures':0,'tables':0,'sections':0,'paragraphs':0,'subsection_bookmarks':0}

    def paragraph(node,style='body'):
        counts['paragraphs']+=1
        return Paragraph(inline(node),styles[style])

    def blocks(node):
        if isinstance(node,str): return []
        if node.tag in {'style','script','nav','footer','hr'}: return []
        if node.tag=='section' and node.attrs.get('id')=='contents': return []
        if node.tag=='h2':
            counts['sections']+=1
            p=paragraph(node,'h2'); p._toc_title=clean(node.text())
            p._bookmark='section-'+str(counts['sections'])
            return ([PageBreak()] if counts['sections']>1 else [])+[p]
        if node.tag=='h3':
            p=paragraph(node,'h3')
            counts['subsection_bookmarks']+=1
            p._toc_title=clean(node.text())
            p._bookmark='subsection-'+str(counts['subsection_bookmarks'])
            p._outline_level=1
            return [p]
        if node.tag=='h4': return [paragraph(node,'h4')]
        if node.tag=='p': return [paragraph(node,'small' if node.attrs.get('class')=='gotcha' else 'body')]
        if node.tag=='div' and node.attrs.get('class')=='readout':
            return [Paragraph(escaped_text(node.text()).replace('\n','<br/>'),styles['readout'])]
        if node.tag=='figure':
            img=node.find('img'); caption=node.find('figcaption')
            data=BytesIO(base64.b64decode(img.attrs['src'].split(',',1)[1]))
            with PILImage.open(data) as source_image:
                rgb=source_image.convert('RGB'); w,h=rgb.size
                png=BytesIO(); rgb.save(png,format='PNG'); png.seek(0)
            scale=min(width/w,174*mm/h)
            illustration=Image(png,width=w*scale,height=h*scale)
            illustration.hAlign='LEFT'; counts['figures']+=1
            cap=paragraph(caption,'caption') if caption else Spacer(1,1)
            # Keep the caption with its figure, but constrain total size below a page.
            return [KeepTogether([Spacer(1,5),illustration,Spacer(1,5),cap])]
        if node.tag=='table':
            rows=[]
            def get_rows(n):
                if isinstance(n,Node):
                    if n.tag=='tr':
                        rows.append([paragraph(c,'cell') for c in n.children
                                     if isinstance(c,Node) and c.tag in {'th','td'}])
                    else:
                        for child in n.children: get_rows(child)
            get_rows(node)
            if not rows: return []
            n=max(map(len,rows)); assert all(len(row)==n for row in rows)
            widths=([.23*width,.12*width,.65*width] if node.attrs.get('class')=='controls'
                    else [width/n]*n)
            table=LongTable(rows,colWidths=widths,repeatRows=1,splitByRow=1,splitInRow=1,
                            hAlign='LEFT',spaceBefore=5,spaceAfter=12)
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef3')),
                ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#ccd7df')),
                ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            counts['tables']+=1; return [table]
        if node.tag in {'ol','ul'}:
            result=[]; num=0
            for li in node.children:
                if not isinstance(li,Node) or li.tag!='li': continue
                num+=1; prefix=f'{num}. ' if node.tag=='ol' else '• '
                body=[c for c in li.children if not isinstance(c,Node) or c.tag not in {'ol','ul'}]
                markup=''.join(inline(c) for c in body)
                result.append(Paragraph(prefix+markup,styles['body']))
                for c in li.children:
                    if isinstance(c,Node) and c.tag in {'ol','ul'}: result.extend(blocks(c))
            return result
        result=[]
        for child in node.children: result.extend(blocks(child))
        return result

    story.extend(blocks(main))
    temporary=output.with_suffix('.pdf.pending')
    doc=ManualDoc(temporary); doc.multiBuild(story)
    from pypdf import PdfReader
    reader=PdfReader(temporary); assert len(reader.pages)>10
    temporary.replace(output)
    return dict(pages=len(reader.pages),**counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path); parser.add_argument('pdf',type=Path)
    args=parser.parse_args(); print(render(args.html,args.pdf))
