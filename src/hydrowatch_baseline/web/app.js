const colors={water_pre:'#3377ff',water_peak:'#00a7c4',flood:'#ed4b3e',receded:'#f4a261'};
const map=L.map('map',{zoomControl:true,attributionControl:false}).setView([51.1,128.2],7);
const groups=Object.fromEntries(Object.keys(colors).map(k=>[k,L.layerGroup().addTo(map)]));
const pairSelect=document.querySelector('#pair');
const statusEl=document.querySelector('#status');

async function loadPairs(){
  const pairs=await fetch('/api/v1/pairs').then(r=>r.json());
  pairSelect.innerHTML=pairs.map(p=>`<option value="${p.pair_id}">${p.aoi_name} · ${p.event_name}</option>`).join('');
}
function setStatus(text,busy=false){statusEl.textContent=text;document.querySelector('#analyze').disabled=busy;}
function area(value){return `${Number(value).toLocaleString('ru-RU')} га`;}
async function analyze(){
  setStatus('Выполняется…',true);
  try{
    const response=await fetch('/api/v1/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pair_id:pairSelect.value})});
    if(!response.ok)throw new Error((await response.json()).detail||'Ошибка анализа');
    const report=await response.json();
    document.querySelector('#pre').textContent=area(report.areas.water_pre_ha);
    document.querySelector('#peak').textContent=area(report.areas.water_peak_ha);
    document.querySelector('#flood').textContent=area(report.areas.flood_ha);
    document.querySelector('#receded').textContent=area(report.areas.receded_ha);
    document.querySelector('#summary').classList.remove('hidden');
    document.querySelector('#json').href=`/api/v1/results/${report.id}/report`;
    document.querySelector('#csv').href=`/api/v1/results/${report.id}/report?format=csv`;
    document.querySelector('#geojson').href=report.contours;
    const geojson=await fetch(report.contours).then(r=>r.json());
    Object.values(groups).forEach(g=>g.clearLayers());
    const layer=L.geoJSON(geojson,{style:f=>({color:colors[f.properties.type],weight:1,fillOpacity:.42}),onEachFeature:(f,l)=>l.bindPopup(`${f.properties.type}: ${f.properties.area_ha} га`)});
    layer.eachLayer(item=>groups[item.feature.properties.type].addLayer(item));
    const bounds=layer.getBounds();if(bounds.isValid())map.fitBounds(bounds.pad(.08));
    setStatus('Отчёт готов');
  }catch(error){setStatus(error.message);}
}
document.querySelector('#analyze').addEventListener('click',analyze);
document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',event=>{const group=groups[event.target.dataset.layer];event.target.checked?group.addTo(map):group.removeFrom(map);}));
loadPairs().catch(error=>setStatus(error.message));
