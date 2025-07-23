document.addEventListener("DOMContentLoaded", function() {
  var contentContainer = document.getElementById('content');
  var content = contentContainer.innerHTML
  var lan = 'e'
  if (document.documentElement.lang === "zh-CN"){
    var pa1 = /\\c(.*?)\\c/g
    var pa2 = /\\e(.*?)\\e/g
  }else{
    var pa1 = /\\e(.*?)\\e/g
    var pa2 = /\\c(.*?)\\c/g
  }
  contentContainer.innerHTML = content.replace(pa1, (match, content)=>{return content}).replace(pa2,'')
});
