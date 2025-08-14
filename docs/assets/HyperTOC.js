const denotes = new Map();

document.addEventListener('DOMContentLoaded', function() {
  var container = document.getElementById('content')
  container.innerHTML = container.innerHTML.replace(/\<!-- ;;([\s\S]*?)\;; -->/gm, (match, content) => {
    content.replace(/\;;;;([\s\S]*?);;;;/g, (m, c) => {
      let arr = c.split('::')
      denotes.set(arr[0], arr[1])
      return c
    })
    return content
  })
});

window.addEventListener('load', function() {
  var box = document.createElement('div')
  box.className = 'flex-container'
  var contentContainer = document.getElementById('content')
  contentContainer.prepend(box)

  var tocElement = document.createElement('div')
  tocElement.className = 'toc'
  const headings = contentContainer.querySelectorAll('h1, h2, h3, h4, h5, h6')
  tocElement.insertAdjacentHTML('afterbegin', '<div class="toc-title">文章目录</div>')
  headings.forEach(heading => {
    if (!heading.id) {
      heading.id = heading.textContent.trim().replace(/\s+/g, '-').toLowerCase()
    }
    const link = document.createElement('a')
    link.href = '#' + heading.id
    link.textContent = heading.textContent
    link.className = 'toc-link'
    link.style.paddingLeft = `${(parseInt(heading.tagName.charAt(1)) - 1) * 10}px`
    tocElement.appendChild(link)
  });
  tocElement.insertAdjacentHTML('beforeend', '<a class="toc-end" onclick="window.scrollTo({top:0,behavior: \'smooth\'});">Top</a>')

  box.appendChild(tocElement)

  var denoteElement = document.createElement('div')
  denoteElement.className = 'denote'
  denoteElement.insertAdjacentHTML('afterbegin', '<div class="denote-title">注释</div>')
  const content = document.createElement('content')
  content.id = 'denote-content'
  content.textContent = '点击文本以显示注释'
  content.className = 'denote-content'
  denoteElement.appendChild(content)

  box.appendChild(tocElement)
  box.appendChild(denoteElement)

  const style = document.createElement('style')
  style.textContent = `
  .flex-container {
    position: fixed;
    display: flex;
    left: 50%;
    transform: translateX(420px);
    overflow-y: auto;
    max-height: 70vh;
    flex-direction: column;
    gap: 15px;
    padding: 10px;
  }
  .toc {
    width: 200px;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 10px;
    overflow-y: auto;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    max-height: 70vh;
  }
  .toc-title{
    font-weight: bold;
    text-align: center;
    border-bottom: 1px solid #ddd;
    padding-bottom: 8px;
  }
  .toc-end{
    font-weight: bold;
    text-align: center;
    visibility: hidden;
  }
  .toc a {
    display: block;
    color: var(--color-diff-blob-addition-num-text);
    text-decoration: none;
    padding: 5px 0;
    font-size: 14px;
    line-height: 1.5;
    border-bottom: 1px solid #e1e4e8;
  }
  .toc a:last-child {
    border-bottom: none;
  }
  .toc a:hover {
    background-color:var(--color-select-menu-tap-focus-bg);
  }
  .denote {
    width: 400px;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 10px;
    overflow-y: auto;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    max-height: 70vh;
  }
  .denote-title{
    font-weight: bold;
    text-align: left;
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
    .flex-container {
      position:static;
      top:auto;
      left:auto;
      transform:none;
      padding:10px;
      margin-bottom:20px;
      max-height: 40vh;
      width:60%;
    }
  }`;
  document.head.appendChild(style)
  window.onscroll = function() {
    const backToTopButton = document.querySelector('.toc-end');
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
      backToTopButton.style = "visibility: visible;"
    } else {
      backToTopButton.style = "visibility: hidden;"
    }
  };
  var elements = document.getElementsByClassName("notranslate");
  for (var i = 0; i < elements.length; i++) {
    item = elements[i]
    const key = item.textContent
    if (denotes.get(key) === undefined){
      continue
    }
    item.addEventListener(
      "click",
      () => {
        const value = denotes.get(key)
        document.getElementsByClassName("denote-title")[0].innerHTML = key
        document.getElementsByClassName('denote-content')[0].innerHTML = value
        if (value.includes('$')) {
          MathJax.typeset()
        }
      },
      false,
    );
  }
});
