const denotes = new Map();

function createDenote() {
  var container = document.getElementById('content')
  container.innerHTML = container.innerHTML.replace(/\\denote{(.*?)}/g, (match, content) => {
    let arr = content.split('::')
    denotes.set(arr[0], arr[1])
    return ''
  })
  var denoteElement = document.createElement('div');
  denoteElement.className = 'denote';
  denotes.forEach((item, i) => {
    console.log(item)
  })
}

window.addEventListener('load', function() {
  createDenote();
  var css = `
  .denote{
    position: fixed;
    top: 130px;
    left: 50%;
    transform: translateX(-100%) translateX(-420px);
    width: 200px;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 10px;
    overflow-y: auto;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    max-height: 70vh;
  }
  .denote-title{

  }
  .denote-content{

  }
  `
  // const style = document.createElement('style');
  // style.textContent = css;
  // document.head.appendChild(style);
});
