function translate(container) {
  if (container === null) {
    return null;
  }
  var content = container.innerHTML
  var lan = 'e'
  if(document.documentElement.lang === 'zh-CN'){
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
  translate(document.getElementById('header').postTitle)
});
