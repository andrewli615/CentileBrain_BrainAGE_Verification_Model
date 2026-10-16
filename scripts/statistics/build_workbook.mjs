import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = process.argv[2];
const data = JSON.parse(await fs.readFile(path.join(root, 'results/summaries/workbook_data.json'), 'utf8'));
const output = path.join(root, 'results/summaries/BrainAGE_method_comparison.xlsx');
const previews = path.join(root, 'results/provenance/workbook_previews');
await fs.mkdir(previews, {recursive:true});
const wb = Workbook.create();
const names = ['Overview', 'Matched statistics', 'Available statistics', 'Training reference', 'Methods and coverage'];
const sheets = Object.fromEntries(names.map(n => [n, wb.worksheets.add(n)]));
const col = n => String.fromCharCode(65+n);
const title = (sheet, text, end) => {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${end}3`).format.font = {name:'Arial', size:10, color:'#20252A'};
  sheet.getRange('A2').values = [[text]];
  sheet.getRange('A2').format.font = {name:'Arial',size:16,bold:true,color:'#174D48'};
  sheet.getRange(`A2:${end}2`).format.rowHeight = 25;
};
function table(sheet, headers, rows, widths, name, start=5) {
  const end = col(headers.length-1), last = start+rows.length;
  sheet.getRange(`A${start}:${end}${last}`).values = [headers,...rows];
  const range = sheet.getRange(`A${start}:${end}${last}`);
  range.format.font = {name:'Arial',size:10,color:'#20252A'};
  range.format.rowHeight = 21;
  range.format.verticalAlignment = 'center';
  range.format.numberFormat = '0.00';
  range.format.horizontalAlignment = 'right';
  widths.forEach((w,i) => {sheet.getRange(`${col(i)}${start}:${col(i)}${last}`).format.columnWidth = w;});
  const nativeTable=sheet.tables.add(`A${start}:${end}${last}`,true,name);
  nativeTable.showFilterButton=true;
  sheet.getRange(`A${start}:${end}${start}`).format = {
    fill:'#245A54',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:38,
  };
  sheet.freezePanes.freezeRows(start);
  return last;
}
const statsKeys = ['Group','Sex','Age_Panel','Method','N','Age_Mean','Age_SD','Age_Min','Age_Max','BrainAGE_Mean','BrainAGE_SD','MAE'];
const statsHeaders = ['Group','Sex','Age panel','Method','N','Age mean (yr)','Age SD (yr)','Age min (yr)','Age max (yr)','BrainAGE mean (yr)','BrainAGE SD (yr)','MAE (yr)'];
for (const [name,key] of [['Matched statistics','matched_statistics'],['Available statistics','available_statistics'],['Training reference','training_reference']]) {
  const sheet = sheets[name];
  title(sheet,name,'L');
  sheet.getRange('A3').values = [[name==='Matched statistics' ? 'Same eligible participants across all five methods within each panel.' : name==='Available statistics' ? 'Each method uses all of its eligible participants. N can differ between methods.' : 'Training/reference controls shown separately. Independence from model training is unverified.']];
  const rows = data[key].map(r=>statsKeys.map(k=>r[k]??null));
  const last = table(sheet,statsHeaders,rows,[17,10,13,48,9,14,13,13,13,20,18,13],key.replaceAll('_',''));
  sheet.getRange(`A6:D${last}`).format.horizontalAlignment='left';
  sheet.getRange(`D6:D${last}`).format.wrapText=true;
  sheet.getRange(`A6:L${last}`).format.rowHeight=32;
  sheet.getRange(`E6:E${last}`).setNumberFormat('#,##0');
}
const overview = sheets.Overview;
title(overview,'BrainAGE method comparison','K');
overview.tabColor='#245A54';
overview.getRange('A3').values=[['Whole-panel MAE in years. Each row uses the same participants across all five methods.']];
const methods = Object.values(data.methods.Methods);
const headers = ['Group','Sex','Age panel','Matched N','Age mean (yr)','Age SD (yr)',
 'Raw\nMAE (yr)','CentileBrain adjusted\nMAE (yr)','Training 10-year\nMAE (yr)','Training 5-year\nMAE (yr)','BrainAGE_S\nMAE (yr)'];
const rows = data.overview.map(r=>[r.Group,r.Sex,r.Age_Panel,r.N,r.Age_Mean,r.Age_SD,...methods.map(m=>r[m])]);
const last = table(overview,headers,rows,[17,9,12,11,13,12,12,19,16,16,14],'MAEOverview');
overview.getRange(`A6:C${last}`).format.horizontalAlignment='left';
overview.getRange(`D6:D${last}`).setNumberFormat('#,##0');
data.overview.forEach((r,i) => {
  const match = data.matched_statistics.findIndex(s=>s.Dataset===r.Dataset&&s.Sex===r.Sex&&s.Age_Panel===r.Age_Panel);
  for(let j=0;j<methods.length;j++) {
    const stat = data.matched_statistics[match+j];
    const n=data.matched_statistics.length+5,r=i+6,m=methods[j].replaceAll('"','""');
    if (stat.MAE!==null) overview.getRange(`${col(j+6)}${r}`).formulas=[[
      `=SUMIFS('Matched statistics'!$L$6:$L$${n},'Matched statistics'!$A$6:$A$${n},$A${r},'Matched statistics'!$B$6:$B$${n},$B${r},'Matched statistics'!$C$6:$C$${n},$C${r},'Matched statistics'!$D$6:$D$${n},"${m}")`
    ]];
  }
  if (i===0 || r.Group!==data.overview[i-1].Group)
    overview.getRange(`A${i+6}:K${i+6}`).format.borders={top:{style:'thin',color:'#829C98'}};
});

const methodology = sheets['Methods and coverage'];
title(methodology,'Methods and coverage','H');
const descriptions = [
 ['Method','Formula / definition','Population / source'],
 ['Raw CentileBrain BAG','CentileBrain predicted brain age - chronological age','CentileBrain output before any adjustment'],
 ['CentileBrain panel-adjusted BAG','Supplied Adjusted_BrainAGE','CentileBrain-provided separate sex-by-age-panel linear adjustment'],
 ['Training-reference 10-year bucket-adjusted BAG','Raw CentileBrain BAG - (training-bin slope * age + intercept)','Sex-specific, panel-specific raw training reference; 10-year bins'],
 ['Training-reference 5-year bucket-adjusted BAG','Raw CentileBrain BAG - (training-bin slope * age + intercept)','Sex-specific, panel-specific raw training reference; 5-year bins'],
 ['BrainAGE_S (cohort age-trend adjusted BAG)','CentileBrain panel-adjusted BAG - cohort slope * (age - cohort mean age)','Own eligible cohort; preserves its mean'],
 ['MAE','Mean absolute BAG across the whole age panel','Not an average of segment MAEs'],
 ['Age panels','5 <= age <= 40; 40 < age <= 90','Internal bins left-closed, right-open; final endpoint included'],
 ['10-year bins','5-15, 15-25, 25-35, 35-40; 40-50 through 80-90','5-year bins start at 5 and 40'],
 ['Unsupported bin','No correction without >=3 training rows and >=2 distinct ages','Unavailable values remain blank'],
 ['Extrapolation','Apply within the bin; flag beyond observed training age support','See participant flags and coverage below'],
 ['Comparison','Shared eligible participants across five methods','Available statistics retain method-specific populations'],
 ['Interpretation','Descriptive adjusted BAG; lower MAE does not establish predictive improvement','The fitted MRI prediction model is not part of this repository'],
];
methodology.getRange('A5:H5').values=[['Method / definition','','','','','','','']];
methodology.getRange('A5:H5').format={fill:'#245A54',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:24};
descriptions.slice(1).forEach((row,i)=>{
  const r=i+6;
  methodology.mergeCells(`A${r}:C${r}`);
  methodology.getRange(`A${r}`).values=[[row[0]]];
  methodology.getRange(`A${r}:C${r}`).format={font:{name:'Arial',size:10,bold:true},wrapText:true,verticalAlignment:'center'};
  methodology.mergeCells(`D${r}:H${r}`);
  methodology.getRange(`D${r}`).values=[[`${row[1]}. ${row[2]}.`]];
  methodology.getRange(`D${r}:H${r}`).format={font:{name:'Arial',size:10},wrapText:true,verticalAlignment:'center'};
  methodology.getRange(`A${r}:H${r}`).format.rowHeight=52;
});
const coverageKeys=['Group','Sex','Age_Panel','Status','N_Input','N_Matched','N_Invalid_Age','N_Unmatched_Source'];
table(methodology,['Group','Sex','Age panel','Status','Input N','Matched N','Invalid age N','Unmatched source N'],
 data.coverage.map(r=>coverageKeys.map(k=>r[k]??null)),[17,10,13,20,12,12,14,18],'Coverage',21);
methodology.getRange('A22:D49').format.horizontalAlignment='left';
methodology.getRange('E22:H49').setNumberFormat('#,##0');
methodology.getRange('A54:H54').format={fill:'#245A54',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:24};
methodology.getRange('A54').values=[['Pinned source files']];
data.sources.filter(r=>r.Path).forEach((r,i)=>{
  const row=55+i;
  methodology.getRange(`A${row}`).values=[[`${r.Dataset} / ${r.Role}`]];
  methodology.mergeCells(`B${row}:H${row}`);
  methodology.getRange(`B${row}`).values=[[`${r.Sex}, ${r.Age_Panel}\n${r.Path}\nSHA-256: ${r.SHA256}`]];
  methodology.getRange(`A${row}:H${row}`).format={font:{name:'Arial',size:10},wrapText:true,verticalAlignment:'center',rowHeight:52};
});
methodology.freezePanes.unfreeze();

// Exercise keyed summary formulas against a changed value and a reordered detail row.
const detail=sheets['Matched statistics'];
const originalFirst=detail.getRange('A6:L6').values;
const finalRow=data.matched_statistics.length+5;
const originalLast=detail.getRange(`A${finalRow}:L${finalRow}`).values;
const originalMAE=overview.getRange('G6').values[0][0];
detail.getRange('L6').values=[[originalFirst[0][11]+1]];
if(Math.abs(overview.getRange('G6').values[0][0]-originalMAE-1)>1e-10) throw new Error('Overview did not recalculate from changed detail');
detail.getRange('A6:L6').values=originalLast;
detail.getRange(`A${finalRow}:L${finalRow}`).values=originalFirst;
if(Math.abs(overview.getRange('G6').values[0][0]-originalMAE)>1e-10) throw new Error('Overview depends on detail row order');
detail.getRange('A6:L6').values=originalFirst;
detail.getRange(`A${finalRow}:L${finalRow}`).values=originalLast;
wb.recalculate();
await fs.writeFile(path.join(root,'results/provenance/workbook_calculation_checks.json'),JSON.stringify({Detail_value_update:true,Detail_row_reordering:true,Temporary_edits_restored:true},null,2));
console.log((await wb.inspect({kind:'table',range:'Overview!A5:K10',include:'values,formulas',tableMaxRows:6,tableMaxCols:11,maxChars:2500})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},maxChars:1000});
await fs.writeFile(path.join(root,'results/provenance/workbook_inspection.ndjson'),errors.ndjson);
for (const name of names) {
  const range=name==='Overview'?`A1:K${last}`:name==='Methods and coverage'?'A1:H17':'A1:L14';
  const blob=await wb.render({sheetName:name,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(previews,name.replaceAll(' ','_')+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
console.log(output);
