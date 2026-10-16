"""Run the bundled spreadsheet author and add native Excel print settings."""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET

from scripts.shared.paths import ROOT, RESULTS


def print_settings(path):
    """Artifact Tool lacks print setup; preserve its workbook and set OOXML metadata."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ET.register_namespace("", ns)
    with ZipFile(path) as z:
        files = {n:z.read(n) for n in z.namelist()}
    # Some runtime exports omit table autoFilter despite showFilterButton=True.
    for name,content in list(files.items()):
        if name.startswith('xl/tables/') and name.endswith('.xml'):
            table=ET.fromstring(content)
            if table.find(f'{{{ns}}}autoFilter') is None:
                table.insert(0,ET.Element(f'{{{ns}}}autoFilter',ref=table.get('ref')))
                files[name]=ET.tostring(table,encoding='utf-8',xml_declaration=True)
    sheet = ET.fromstring(files["xl/worksheets/sheet1.xml"])
    props = sheet.find(f"{{{ns}}}sheetPr")
    if props is None:
        props = ET.Element(f"{{{ns}}}sheetPr")
        sheet.insert(0,props)
    setup = props.find(f"{{{ns}}}pageSetUpPr")
    if setup is None:
        setup = ET.SubElement(props,f"{{{ns}}}pageSetUpPr")
    setup.set("fitToPage","1")
    # Column/row freeze combined in one pane, irrespective of exporter defaults.
    view = sheet.find(f"{{{ns}}}sheetViews/{{{ns}}}sheetView")
    pane = view.find(f"{{{ns}}}pane")
    if pane is not None:
        pane.attrib.update(xSplit="3",ySplit="5",topLeftCell="D6",activePane="bottomRight",state="frozen")
    page = sheet.find(f"{{{ns}}}pageSetup")
    if page is None:
        page = ET.Element(f"{{{ns}}}pageSetup")
        # pageSetup follows margins and precedes header/footer, drawings, and tableParts.
        before = next((i for i,c in enumerate(sheet) if c.tag.split('}')[-1] in ('headerFooter','rowBreaks','colBreaks','drawing','legacyDrawing','tableParts','extLst')),len(sheet))
        sheet.insert(before,page)
    page.attrib.update(orientation="landscape",paperSize="9",fitToWidth="1",fitToHeight="1")
    files["xl/worksheets/sheet1.xml"] = ET.tostring(sheet,encoding="utf-8",xml_declaration=True)
    book=ET.fromstring(files["xl/workbook.xml"])
    definitions=book.find(f"{{{ns}}}definedNames")
    if definitions is None:
        definitions=ET.Element(f"{{{ns}}}definedNames")
        idx=next((i for i,c in enumerate(book) if c.tag.split('}')[-1] in ('calcPr','extLst')),len(book))
        book.insert(idx,definitions)
    last=len(json.loads((RESULTS/'summaries/workbook_data.json').read_text())['overview'])+5
    ET.SubElement(definitions,f"{{{ns}}}definedName",name="_xlnm.Print_Area",localSheetId="0").text=f"'Overview'!$A$1:$K${last}"
    files["xl/workbook.xml"]=ET.tostring(book,encoding="utf-8",xml_declaration=True)
    with ZipFile(path,"w",ZIP_DEFLATED) as z:
        for name,content in files.items():
            z.writestr(name,content)


def build_workbook():
    bundled = Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies"
    node = Path(os.environ.get("BRAINAGE_NODE",str(bundled/"node/bin/node")))
    modules = Path(os.environ.get("BRAINAGE_NODE_MODULES",str(bundled/"node/node_modules")))
    if not node.is_file() or not modules.is_dir():
        raise RuntimeError("Spreadsheet runtime missing. Set BRAINAGE_NODE and BRAINAGE_NODE_MODULES; CSV summaries are already saved.")
    with tempfile.TemporaryDirectory(prefix="brainage-workbook-") as tmp:
        directory=Path(tmp)
        (directory/"node_modules").symlink_to(modules,target_is_directory=True)
        shutil.copy2(Path(__file__).with_name("build_workbook.mjs"),directory/"build.mjs")
        subprocess.run([str(node),str(directory/"build.mjs"),str(ROOT)],check=True)
    print_settings(RESULTS/"summaries/BrainAGE_method_comparison.xlsx")
    inspection=RESULTS/"summaries/BrainAGE_method_comparison.xlsx.inspect.ndjson"
    if inspection.exists():
        shutil.move(inspection,RESULTS/"provenance/workbook_export_inspection.ndjson")
