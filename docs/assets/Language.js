function transStr(str, force_cn){
  if (force_cn || navigator.language=== 'zh-CN') {
    var pa1 = /\\c(.*?)\\c/g
    var pa2 = /\\e(.*?)\\e/g
  }else{
    var pa1 = /\\e(.*?)\\e/g
    var pa2 = /\\c(.*?)\\c/g
  }
  return str.replace(pa1, (match, content)=>{return content}).replace(pa2,'')
}

function translate(container, force_cn){
  if (container === null) {
    return null;
  }
  container.innerHTML = transStr(container.innerHTML, force_cn)
}

document.addEventListener('DOMContentLoaded', function(){
  var elements = document.getElementsByClassName('markdown-alert markdown-alert-note')
  var force_cn = false
  if (elements.length > 0 && elements[0].innerHTML.includes('This article currently only supports Chinese.')){
    force_cn = true
  }
  document.title = transStr(document.title)
  translate(document.getElementById('listTitle'), force_cn)
  translate(document.getElementById('content'), force_cn)
  elements = document.getElementsByClassName('postTitle')
  if (elements.length > 0) {
    translate(elements[0], force_cn)
  }
});
