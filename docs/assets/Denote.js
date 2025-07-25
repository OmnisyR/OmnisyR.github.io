const denotes = new Map();

document.addEventListener('DOMContentLoaded', function() {
  var container = document.getElementById('content')
  container.innerHTML = container.innerHTML.replace(/\\denote{(.*?)}/g, (match, content) => {
    let arr = content.split('::')
    denotes.set(arr[0], arr[1])
    return ''
  })
});

window.addEventListener('load', function() {
  var denoteElement = document.createElement('div')
  denoteElement.className = 'denote'
  var container = document.getElementById('content')
  var elements = document.getElementsByClassName('toc')
  var hasTOC = elements.length > 0
  document.getElementById('content').prepend(denoteElement);
  denoteElement.insertAdjacentHTML('afterbegin', '<div class="denote-title">注释</div>')
  const content = document.createElement('content')
  content.textContent = '点击文本以显示注释'
  content.className = 'denote-content'
  denoteElement.appendChild(content)
  const style = document.createElement('style')
  style.textContent = `
  .denote{
    position: fixed;
    top: 130px;
    left: 50%;
    transform: translateX(50%) translateX(320px);
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
  .denote content{
    display: block;
    color: var(--color-diff-blob-addition-num-text);
    text-decoration: none;
    padding: 5px 0;
    font-size: 14px;
    line-height: 1.5;
    border-bottom: 1px solid #e1e4e8;
  }
  @media (max-width: 1249px) {
    .denote{
      position:static;
      top:auto;
      left:auto;
      transform:none;
      padding:10px;
      margin-bottom:20px;
      max-height: 40vh;
      width:60%;
    }
  }
  `;
  document.head.appendChild(style);
});
