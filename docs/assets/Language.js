function translate(container) {
  if (container === null) {
    return null;
  }
  var content = container.innerHTML
  if (navigator.language=== 'zh-CN') {
    var pa1 = /\\c(.*?)\\c/g
    var pa2 = /\\e(.*?)\\e/g
  }else{
    var pa1 = /\\e(.*?)\\e/g
    var pa2 = /\\c(.*?)\\c/g
  }
  container.innerHTML = content.replace(pa1, (match, content)=>{return content}).replace(pa2,'')
}

document.addEventListener('DOMContentLoaded', function() {
  translate(document.getElementById('listTitle'))
  translate(document.getElementById('content'))
  let elements = document.getElementsByClassName('postTitle')
  if (elements.length > 0) {
    translate(elements[0])
  }
});
