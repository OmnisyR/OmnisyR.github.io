const denotes = new Map();

window.addEventListener('load', function() {
  var container = document.getElementById('content')
  container.innerHTML = container.innerHTML.replace(/\\denote{(.*?)}/g, (match, content) => {
    let arr = content.split('::')
    denotes.set(arr[0], arr[1])
    return ''
  })
  var denoteElement = document.createElement('div');
  denoteElement.className = 'denote';
  container.prepend(denoteElement);
  denoteElement.insertAdjacentHTML('afterbegin', '<div class="denote-title">注释</div>')
  const style = document.createElement('style');
  style.textContent =  `
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
    font-weight: bold;
    text-align: center;
    border-bottom: 1px solid #ddd;
    padding-bottom: 8px;
  }
  .denote-content{

  }
  `;
  document.head.appendChild(style);
  // const style = document.createElement('style');
  // style.textContent = css;
  // document.head.appendChild(style);
});
