'use strict';
const $ = id => document.getElementById(id);
let selectedFile, previewUrl, busy = false;
const el = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
function message(id,text,error=false){const n=$(id);n.textContent=text;n.className=error?'error':'';}
async function api(path,options={}){
  const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),120000);
  try {const response=await fetch(path,{...options,signal:controller.signal});let data;try{data=await response.json();}catch{throw new Error('The service returned an unexpected response. Please retry.');}
    if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Check your input and try again.');return data;
  } catch(e){if(e.name==='AbortError')throw new Error('The request timed out. Please try again.');if(e instanceof TypeError)throw new Error('Cannot reach the server. Check your connection and retry.');throw e;}finally{clearTimeout(timer);}
}
function setFile(file){
  message('analyze-message','');
  $('result-count').textContent='';
  $('analysis-results').replaceChildren(el('div','Choose an image and click Analyze to read the prescription.','empty'));
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  selectedFile=undefined;$('preview-wrap').hidden=true;$('analyze-button').disabled=true;$('preview').removeAttribute('src');
  if(!file)return;
  if(!['image/jpeg','image/png','image/webp','image/gif'].includes(file.type))return message('analyze-message','Choose a JPEG, PNG, WebP or GIF image.',true);
  if(file.size>5*1024*1024)return message('analyze-message','Image must be 5 MB or smaller.',true);
  selectedFile=file;previewUrl=URL.createObjectURL(file);$('preview').src=previewUrl;$('preview-wrap').hidden=false;$('file-info').textContent=`${file.name} · ${(file.size/1024).toFixed(0)} KB`;$('analyze-button').disabled=busy;
}
$('image').addEventListener('change',()=>setFile($('image').files[0]));
$('remove-image').addEventListener('click',()=>{$('image').value='';setFile();});
$('sample-button').addEventListener('click',()=>{
  // A generated fictional document, containing no patient data.
  const canvas=document.createElement('canvas');canvas.width=1000;canvas.height=700;const ctx=canvas.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,1000,700);ctx.fillStyle='#193441';ctx.font='bold 36px sans-serif';ctx.fillText('FICTIONAL PRESCRIPTION — DEMO ONLY',60,90);ctx.font='26px sans-serif';ctx.fillText('No real patient. Not valid for treatment.',60,145);ctx.font='34px sans-serif';ctx.fillText('Rx: Panadol 500 mg',60,265);ctx.font='28px sans-serif';ctx.fillText('Frequency: once daily',60,335);ctx.fillText('Route: oral',60,390);ctx.fillText('Duration: 3 days',60,445);ctx.font='22px sans-serif';ctx.fillText('For software testing only.',60,590);
  canvas.toBlob(blob=>{$('image').value='';setFile(new File([blob],'fictional-prescription.png',{type:'image/png'}));message('analyze-message','Fictional sample ready. Click Analyze to send it to Gemini.');},'image/png');
});
function card(medicine,lookup=false){
  const matching=medicine.matching||{};const matches=matching.matches||[];const uncertain=!matches.length||['LOW','MEDIUM','NOT_FOUND'].includes(matching.confidence)||medicine.read_confidence==='low'||medicine.read_confidence==='medium';
  const article=el('article',undefined,'medicine');const head=el('div',undefined,'medicine-head');const title=el('div');title.append(el('div',lookup?'YOUR SEARCH':'OCR ACTUALLY READ','small-label'),el('h3',medicine.ocr_name||'Unreadable medicine name'));head.append(title,el('span',!matches.length?'No database match':uncertain?'Needs review':'Database suggestion',`badge${uncertain?' uncertain':''}`));article.append(head);
  if(!lookup){const dl=el('dl',undefined,'details');for(const key of ['dose','frequency','route','duration']){const item=el('div');item.append(el('dt',key[0].toUpperCase()+key.slice(1)),el('dd',medicine[key]||'Not read'));dl.append(item);}article.append(dl);article.append(el('p',`Reading confidence: ${medicine.read_confidence||'Not reported'}. Matching confidence: ${matching.confidence||'Not reported'}.`));}
  if(medicine.corrected_from)article.append(el('p',`Name used for matching: ${medicine.cleaned_name}. Original reading is shown above.`));
  if(medicine.notes)article.append(el('p',medicine.notes));
  const suggestions=el('div',undefined,'suggestions');suggestions.append(el('div','SUGGESTED DATABASE MATCHES','small-label'));
  if(matches.length){const list=el('ul');for(const match of matches)list.append(el('li',match));suggestions.append(list);}else suggestions.append(el('p','No match found. Keep this entry for manual review.'));
  if(matching.note)suggestions.append(el('p',matching.note));article.append(suggestions);return article;
}
$('lookup-form').addEventListener('submit',async event=>{event.preventDefault();const query=$('query').value.trim();if(!query)return;const button=$('lookup-button');button.disabled=true;button.textContent='Searching…';$('lookup-results').replaceChildren();message('lookup-message','Searching the medicine database…');try{const data=await api('/match',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({names:[query],k:5})});const entries=Object.entries(data.results||{});message('lookup-message',entries.length?'Search complete. Review suggestions below.':'No results returned.');for(const [name,matching] of entries)$('lookup-results').append(card({ocr_name:name,matching},true));}catch(e){message('lookup-message',e.message,true);}finally{button.disabled=false;button.textContent='Search';}});
$('analyze-form').addEventListener('submit',async event=>{event.preventDefault();if(!selectedFile||busy)return;busy=true;$('analyze-button').disabled=true;$('analyze-button').textContent='Analyzing…';$('sample-button').disabled=true;$('remove-image').disabled=true;$('image').disabled=true;$('result-count').textContent='';$('analysis-results').replaceChildren(el('div','Gemini is reading the fictional prescription. This may take a moment.','empty'));message('analyze-message','Uploading to Gemini and looking up database suggestions…');try{const form=new FormData();form.append('file',selectedFile);const data=await api('/analyze',{method:'POST',body:form});const meds=data.medicines||[];$('analysis-results').replaceChildren();$('result-count').textContent=`${meds.length} ${meds.length===1?'entry':'entries'}`;if(!meds.length)$('analysis-results').append(el('div','No medicines detected. Try a clearer image or use text lookup.','empty'));for(const med of meds)$('analysis-results').append(card(med));message('analyze-message','Analysis complete. Check every original reading and suggested match.');}catch(e){message('analyze-message',e.message,true);$('analysis-results').replaceChildren(el('div','Analysis could not be completed. You can retry or use text lookup.','empty'));}finally{busy=false;$('analyze-button').disabled=!selectedFile;$('analyze-button').textContent='Analyze prescription';$('sample-button').disabled=false;$('remove-image').disabled=false;$('image').disabled=false;}});
api('/health').then(data=>{$('ocr-status').textContent=data.ocr_configured?'Live OCR is configured. Use fictional images only.':'Live OCR pending server key setup. Text lookup is available.';}).catch(()=>{$('ocr-status').textContent='Service is starting or unavailable. Please retry shortly.';});
