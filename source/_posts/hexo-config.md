---
title: Hexo搭建个性化blog
date: 2018-04-30 22:04:07
categories: 教程
tags: 
- Hexo
- Blog
top: 10
---
[Hexo][]是高效的静态站点生成框架，基于 Node.js。 通过 Hexo 你可以轻松地使用 Markdown 编写文章，除了 Markdown 本身的语法之外，还可以使用 Hexo 提供的 标签插件 来快速的插入特定形式的内容。在这篇文章中，假定你已经成功安装了 Hexo，并使用 Hexo 提供的命令创建了一个站点。

[Hexo]: https://hexo.io/zh-cn/

<h3 id="overview">初级搭建</h3>

<h4 id="install">安装</h4>

在安装前，您必须检查电脑中是否已安装下列应用程序：

- [Node.js][]
- [Git][]

[Node.js]:https://nodejs.org/en/
[Git]:https://git-scm.com/

1、node.js安装

ubuntu:

    for Node.js 10:
        curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
        sudo apt-get install -y nodejs
    for Node.js 10:
        curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
        sudo apt-get install -y nodejs

mac:

    brew install node

2、git安装

ubuntu:

    sudo apt-get install git-core

mac:

    brew install git

3、hexo安装

使用 npm 即可完成 Hexo 的安装

    npm install -g hexo-cli

***

<h4 id="blog">搭建博客</h4>

1、初始化

    hexo init + 需要初始化的目录

2、安装依赖

进入初始化后的目录，并安装依赖

    npm install

3、生成页面文件

    hexo generate # 简写 hexo g

4、启动本地服务

    hexo server # 简写 hexo s 调试加参数：--debug

5、 测试

    浏览器输入： http://localhost:4000

6、安装主题

安装[Next][]主题，访问更多[主题][]

[Next]:http://theme-next.iissnan.com/theme-settings.html
[主题]:https://hexo.io/themes/

    git clone https://github.com/iissnan/hexo-theme-next themes/next

根目录下`_config.yml`，修改theme值

    theme: next # 刚刚安装的主题名称

***
<h4 id="commond">常用命令</h4>

Hexo 安装升级

    npm install hexo -g #安装
    npm update hexo -g #升级
    hexo init #初始化

常用简写

    hexo n "我的博客" == hexo new "我的博客" #新建文章
    hexo p == hexo publish
    hexo g == hexo generate#生成
    hexo s == hexo server #启动服务预览
    hexo d == hexo deploy#部署

启动本地服务

    hexo server #Hexo #会监视文件变动并自动更新，您无须重启服务器。
    hexo server -s #静态模式
    hexo server -p 5000 #更改端口
    hexo server -i 192.168.1.1 #自定义 IP

监视文件变动

    hexo generate #使用 Hexo 生成静态文件快速而且简单
    hexo generate --watch #监视文件变动
    hexo clean #清除缓存 网页正常情况下可以忽略此条命令
部署

    #两个命令的作用是相同的
    hexo generate --deploy
    hexo deploy --generate
    hexo deploy -g
    hexo server -g

草稿

    # 新建草稿
    hexo new draft <title>
    # 发布草稿为post
    hexo publish draft <title>

模板

    hexo new "postName" #新建文章
    hexo new page "pageName" #新建页面
    hexo generate #生成静态页面至public目录
    hexo server #开启预览访问端口（默认端口4000，'ctrl + c'关闭server）
    hexo deploy #将.deploy目录部署到GitHub
    hexo new [layout] <title>
    hexo new photo "My Gallery"
    hexo new "Hello World" --lang tw

写作时间

    变量    描述
    :title    标题
    :year    建立的年份（4 位数）
    :month    建立的月份（2 位数）
    :i_month    建立的月份（去掉开头的零）
    :day    建立的日期（2 位数）
    :i_day    建立的日期（去掉开头的零）

***

<h3 id="individual">个性化设置</h3>

<h4 id="fork">实现fork me on github</h4>

点击[这里](https://blog.github.com/2008-12-19-github-ribbons/)或[Here](http://tholman.com/github-corners/)，挑选喜欢的样式，复制其中的代码，例如：

    <a href="https://github.com/you">
    <img style="position: absolute; top: 0; left: 0; border: 0;" src="https://s3.amazonaws.com/github/ribbons/forkme_left_red_aa0000.png" alt="Fork me on GitHub">
    </a>

然后粘贴刚才复制的代码到`themes/next/layout/_layout.swig`文件中(放在`<div class="headband"></div>`的下面)，并把`href`改为你的github地址。

***
<h4 id="fontSize">字体大小</h4>

打开`\themes\next\source\css\ _variables\base.styl`文件，修改font-size-base，如下所示：

    font-size-base = 16px

***

<h4 id="rss">添加RSS</h4>

根目录下安装插件：

    npm install --save hexo-generator-feed

配置`主题配置文件`下的rss:

    rss: /atom.xml #或者为空

重新生成静态文件，将在`public`下生成`atom.xml`

    hexo g

***

<h4 id="background">添加动态线条背景</h4>

<p id='step1'>1、修改_layout.swig</p>

打开 `next/layout/_layout.swig`,在 `< /body>`之前添加代码(注意不要放在`< /head>`的后面)

```
    {% if theme.canvas_nest %}
    <script type="text/javascript" src="//cdn.bootcss.com/canvas-nest.js/1.0.0/canvas-nest.min.js">
    </script>
    {% endif %}
```

2、修改配置文件
打开 `/next/_config.yml`,在里面添加如下代码：(可以放在最后面)

    # --------------------------------------------------------------
    # background settings
    # --------------------------------------------------------------
    # add canvas-nest effect
    # see detail from https://github.com/hustcc/canvas-nest.js
    canvas_nest: true

3、重新生成文件

    hexo clean
    hexo g
    hexo s

4、配置线条属性

替换[第一步](#step1)中的代码：

```
{% if theme.canvas_nest %}
<script type="text/javascript"
color="0,0,255" opacity='0.7' zIndex="-2" count="99" src="//cdn.bootcss.com/canvas-nest.js/1.0.0/canvas-nest.min.js"></script>
{% endif %}
```

配置项说明

- color ：线条颜色, 默认: '0,0,0'；三个数字分别为(R,G,B)
- opacity: 线条透明度（0~1）, 默认: 0.5
- count: 线条的总数量, 默认: 150
- zIndex: 背景的z-index属性，css属性用于控制所在层的位置, 默认: -1

***
<h4 id="click">点击出桃心</h4>

新建`love.js`文件，放入路径`/themes/next/source/js/src`

```
!function(e,t,a){function n(){c(".heart{width: 10px;height: 10px;position: fixed;background: #f00;transform: rotate(45deg);-webkit-transform: rotate(45deg);-moz-transform: rotate(45deg);}.heart:after,.heart:before{content: '';width: inherit;height: inherit;background: inherit;border-radius: 50%;-webkit-border-radius: 50%;-moz-border-radius: 50%;position: fixed;}.heart:after{top: -5px;}.heart:before{left: -5px;}"),o(),r()}function r(){for(var e=0;e<d.length;e++)d[e].alpha<=0?(t.body.removeChild(d[e].el),d.splice(e,1)):(d[e].y--,d[e].scale+=.004,d[e].alpha-=.013,d[e].el.style.cssText="left:"+d[e].x+"px;top:"+d[e].y+"px;opacity:"+d[e].alpha+";transform:scale("+d[e].scale+","+d[e].scale+") rotate(45deg);background:"+d[e].color+";z-index:99999");requestAnimationFrame(r)}function o(){var t="function"==typeof e.onclick&&e.onclick;e.onclick=function(e){t&&t(),i(e)}}function i(e){var a=t.createElement("div");a.className="heart",d.push({el:a,x:e.clientX-5,y:e.clientY-5,scale:1,alpha:1,color:s()}),t.body.appendChild(a)}function c(e){var a=t.createElement("style");a.type="text/css";try{a.appendChild(t.createTextNode(e))}catch(t){a.styleSheet.cssText=e}t.getElementsByTagName("head")[0].appendChild(a)}function s(){return"rgb("+~~(255*Math.random())+","+~~(255*Math.random())+","+~~(255*Math.random())+")"}var d=[];e.requestAnimationFrame=function(){return e.requestAnimationFrame||e.webkitRequestAnimationFrame||e.mozRequestAnimationFrame||e.oRequestAnimationFrame||e.msRequestAnimationFrame||function(e){setTimeout(e,1e3/60)}}(),n()}(window,document);
```

打开`\themes\next\layout\_layout.swig`文件，添加如下代码：

    <!-- 页面点击小红心 -->
    <script type="text/javascript" src="/js/src/love.js"></script>

***
<h4 id="link">修改文章内链接文本样式</h4>

修改文件 `themes\next\source\css\_common\components\post\post.styl`，在末尾添加如下css样式:

    // 文章内链接文本样式
    .post-body p a{
    color: #0593d3;
    border-bottom: none;
    border-bottom: 1px solid #0593d3;
    &:hover {
        color: #fc6423;
        border-bottom: none;
        border-bottom: 1px solid #fc6423;
    }
    }

其中选择`.post-body`是为了不影响标题，选择 `p`是为了不影响首页“阅读全文”的显示样式,颜色可以自己定义。

***

<h4 id="tag_icon">修改标签显示</h4>

修改模板`/themes/next/layout/_macro/post.swig`，搜索 `rel="tag">#`，将 `#`换成`<i class="fa fa-tag"></i>`

***

<h4 id="endtag">添加“本文结束”标记</h4>

在路径`\themes\next\layout\_macro`中新建 `passage-end-tag.swig` 文件,并添加以下内容：

```
    <div>
        {% if not is_index %}
            <div style="text-align:center;color: #ccc;font-size:14px;">
            -------------本文结束<i class="fa fa-paw"></i>感谢您的阅读-------------</div>
        {% endif %}
    </div>
```

打开`\themes\next\layout\_macro\post.swig`文件，在`post-body` 之后， `post-footer` 之前添加div:

```
    <div>
    {% if not is_index %}
        {% include 'passage-end-tag.swig' %}
    {% endif %}
    </div>
```

打开主题配置文件`_config.yml`,在末尾添加：

    # 文章末尾添加“本文结束”标记
    passage_end_tag:
      enabled: true

***
<h4 id="rotate">头像旋转</h4>

打开`\themes\next\source\css\_common\components\sidebar\sidebar-author.styl`，在里面添加如下代码：

    .site-author-image {
    display: block;
    margin: 0 auto;
    padding: $site-author-image-padding;
    max-width: $site-author-image-width;
    height: $site-author-image-height;
    border: $site-author-image-border-width solid $site-author-image-border-color;

    /* 头像圆形 */
    border-radius: 80px;
    -webkit-border-radius: 80px;
    -moz-border-radius: 80px;
    box-shadow: inset 0 -1px 0 #333sf;

    /* 设置循环动画 [animation: (play)动画名称 (2s)动画播放时长单位秒或微秒 (ase-out)动画播放的速度曲线为以低速结束
        (1s)等待1秒然后开始动画 (1)动画播放次数(infinite为循环播放) ]*/

    /* 鼠标经过头像旋转360度 */
    -webkit-transition: -webkit-transform 1.0s ease-out;
    -moz-transition: -moz-transform 1.0s ease-out;
    transition: transform 1.0s ease-out;
    }

    img:hover {
    /* 鼠标经过停止头像旋转
    -webkit-animation-play-state:paused;
    animation-play-state:paused;*/

    /* 鼠标经过头像旋转360度 */
    -webkit-transform: rotateZ(360deg);
    -moz-transform: rotateZ(360deg);
    transform: rotateZ(360deg);
    }

    /* Z 轴旋转动画 */
    @-webkit-keyframes play {
    0% {
        -webkit-transform: rotateZ(0deg);
    }
    100% {
        -webkit-transform: rotateZ(-360deg);
    }
    }
    @-moz-keyframes play {
    0% {
        -moz-transform: rotateZ(0deg);
    }
    100% {
        -moz-transform: rotateZ(-360deg);
    }
    }
    @keyframes play {
    0% {
        transform: rotateZ(0deg);
    }
    100% {
        transform: rotateZ(-360deg);
    }
    }

***

<h4 id="gulp">博文压缩</h4>

在根目录下执行命令：

    npm install gulp -g
    npm install gulp-minify-css gulp-uglify gulp-htmlmin gulp-htmlclean gulp --save

`根目录`下新建`gulpfile.js`，加入代码:

    var gulp = require('gulp');
    var minifycss = require('gulp-minify-css');
    var uglify = require('gulp-uglify');
    var htmlmin = require('gulp-htmlmin');
    var htmlclean = require('gulp-htmlclean');
    // 压缩 public 目录 css
    gulp.task('minify-css', function() {
        return gulp.src('./public/**/*.css')
            .pipe(minifycss())
            .pipe(gulp.dest('./public'));
    });
    // 压缩 public 目录 html
    gulp.task('minify-html', function() {
    return gulp.src('./public/**/*.html')
        .pipe(htmlclean())
        .pipe(htmlmin({
            removeComments: true,
            minifyJS: true,
            minifyCSS: true,
            minifyURLs: true,
        }))
        .pipe(gulp.dest('./public'))
    });
    // 压缩 public/js 目录 js
    gulp.task('minify-js', function() {
        return gulp.src('./public/**/*.js')
            .pipe(uglify())
            .pipe(gulp.dest('./public'));
    });
    // 执行 gulp 命令时执行的任务
    gulp.task('default', [
        'minify-html','minify-css','minify-js'
    ]);

生成博文是执行 `hexo g && gulp` 就会根据 gulpfile.js 中的配置，对 public 目录中的静态资源文件进行压缩。

***
<h4 id="codestyle">代码块样式</h4>

打开`\themes\next\source\css\_custom\custom.styl`,加入代码：

    // Custom styles.
    code {
        color: #ff7600;
        background: #fbf7f8;
        margin: 2px;
    }
    // 大代码块的自定义样式
    .highlight, pre {
        margin: 5px 0;
        padding: 5px;
        border-radius: 3px;
    }
    .highlight, code, pre {
        border: 1px solid #d6d6d6;
    }

***
<h4 id="social_icon">社交小图标</h4>

打开主题配置文件`_config.yml`，搜索`social_icons:`,在图标库找自己喜欢的小图标，并将名字复制在对应位置

***

<h4 id="shadow">主页添加阴影效果</h4>

打开`\themes\next\source\css\_custom\custom.styl,`向里面加入：

    // 主页文章添加阴影效果
    .post {
    margin-top: 60px;
    margin-bottom: 60px;
    padding: 25px;
    -webkit-box-shadow: 0 0 5px rgba(202, 203, 203, .5);
    -moz-box-shadow: 0 0 5px rgba(202, 203, 204, .5);
    }

***

<h4 id="visit">添加访问量</h4>

打开`\themes\next\layout\_partials\footer.swig`文件,在`copyright`代码块前添加：

    <script async src="https://dn-lbstatics.qbox.me/busuanzi/2.3/busuanzi.pure.mini.js"></script>

打开`\themes\next\layout\_macro\post.swig`文件，在需要显示的地方添加：

    <div class="powered-by">
    <i class="fa fa-user-md"></i><span id="busuanzi_container_site_uv">
    本站访客数:<span id="busuanzi_value_site_uv"></span>
    </span>
    </div>

两中不同计算方式的统计代码：

- pv的方式，单个用户连续点击n篇文章，记录n次访问量
    ```
    <span id="busuanzi_container_site_pv">
        本站总访问量<span id="busuanzi_value_site_pv"></span>次
    </span>
    ```

- uv的方式，单个用户连续点击n篇文章，只记录1次访客数
    ```
    <span id="busuanzi_container_site_uv">
    本站总访问量<span id="busuanzi_value_site_uv"></span>次
    </span>
    ```

添加之后再执行hexo d -g，然后再刷新页面。

***
<h4 id="hot">添加热度</h4>

打开`/themes/next/layout/_macro/post.swig`，搜索`leancloud-visitors-count`，修改对应热度显示文字。

打开`/themes/next/languages/zh-Hans.yml`,将`visitors`值改为“热度”。

***
<h4 id="wordcount">字数统计</h4>

在根目录下，安装计数插件：

    npm install hexo-wordcount --save

若要在网站底部显示，可以在文件`/themes/next/layout/_partials/footer.swig`尾部加入代码：

    <div class="theme-info">
    <div class="powered-by"></div>
    <span class="post-count">博客全站共{{ totalcount(site) }}字</span>
    </div>

***

<h4 id="readme">添加READEME.md</h4>

每个项目下一般都有一个 `README.md` 文件，但是使用 hexo 部署到仓库后，项目下是没有 `README.md` 文件的。

在 Hexo 目录下的 `source` 根目录下添加一个 `README.md` 文件，修改站点配置文件 `_config.yml`，将 `skip_render` 参数的值设置如下：

    skip_render: README.md

`使用 hexo d` 命令部署博客的时候就不会在渲染 `README.md` 文件。

***
<h4 id="favicon">网站图标</h4>

在[EasyIcon](http://www.easyicon.net/)中找一张（32*32）的`ico`图标,或者去别的网站下载或者制作，并将图标名称改为`favicon.ico``，然后把图标放在/themes/next/source/images`里，并且修改主题配置文件：

    # Put your favicon.ico into `hexo-site/source/` directory.
    favicon: /favicon.ico

***

<h4 id="static">统计功能</h4>

在根目录下安装 hexo-wordcount,运行：

    npm install hexo-wordcount --save

然后在主题的配置文件中，配置如下：

    # Post wordcount display settings
    # Dependencies: https://github.com/willin/hexo-wordcount
    post_wordcount:
    item_text: true
    wordcount: true
    min2read: true

***
<h4 id="loading">进度条</h4>

打开`/themes/next/layout/_partials/head.swig`文件，添加代码:

    <script src="//cdn.bootcss.com/pace/1.0.2/pace.min.js"></script>
    <link href="//cdn.bootcss.com/pace/1.0.2/themes/pink/pace-theme-flash.css" rel="stylesheet">

默认的是粉色的，要改变颜色可以在`/themes/next/layout/_partials/head.swig`文件中添加如下代码（接在刚才link的后面）:

    <style>
        .pace .pace-progress {
            background: #1E92FB; /*进度条颜色*/
            height: 3px;
        }
        .pace .pace-progress-inner {
            box-shadow: 0 0 10px #1E92FB, 0 0 5px     #1E92FB; /*阴影颜色*/
        }
        .pace .pace-activity {
            border-top-color: #1E92FB;    /*上边框颜色*/
            border-left-color: #1E92FB;    /*左边框颜色*/
        }
    </style>

升级最新版的next主题，只需修改主题配置文件`_config.yml`将`pace: false`改为`pace: true`即可，还可以换不同样式的加载条。

    pace: true
    pace_theme: pace-theme-minimal

***
<h4 id="info">版权信息</h4>

在目录 `next/layout/_macro/`下添加 `my-copyright.swig`：

```
    {% if page.copyright %}
    <div class="my_post_copyright">
    <script src="//cdn.bootcss.com/clipboard.js/1.5.10/clipboard.min.js"></script>

    <!-- JS库 sweetalert 可修改路径 -->
    <script src="https://cdn.bootcss.com/jquery/2.0.0/jquery.min.js"></script>
    <script src="https://unpkg.com/sweetalert/dist/sweetalert.min.js"></script>
    <p><span>本文标题:</span><a href="{{ url_for(page.path) }}">{{ page.title }}</a></p>
    <p><span>文章作者:</span><a href="/" title="访问 {{ theme.author }} 的个人博客">{{ theme.author }}</a></p>
    <p><span>发布时间:</span>{{ page.date.format("YYYY年MM月DD日 - HH:MM") }}</p>
    <p><span>最后更新:</span>{{ page.updated.format("YYYY年MM月DD日 - HH:MM") }}</p>
    <p><span>原始链接:</span><a href="{{ url_for(page.path) }}" title="{{ page.title }}">{{ page.permalink }}</a>
        <span class="copy-path"  title="点击复制文章链接"><i class="fa fa-clipboard" data-clipboard-text="{{ page.permalink }}"  aria-label="复制成功！"></i></span>
    </p>
    <p><span>许可协议:</span><i class="fa fa-creative-commons"></i> <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" title="Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)">署名-非商业性使用-禁止演绎 4.0 国际</a> 转载请保留原文链接及作者。</p>
    </div>
    <script>
        var clipboard = new Clipboard('.fa-clipboard');
        $(".fa-clipboard").click(function(){
        clipboard.on('success', function(){
            swal({
            title: "",
            text: '复制成功',
            icon: "success",
            showConfirmButton: true
            });
            });
        });
    </script>
    {% endif %}
```

在目录`next/source/css/_common/components/post/`下添加`my-post-copyright.styl`：

    .my_post_copyright {
    width: 85%;
    max-width: 45em;
    margin: 2.8em auto 0;
    padding: 0.5em 1.0em;
    border: 1px solid #d3d3d3;
    font-size: 0.93rem;
    line-height: 1.6em;
    word-break: break-all;
    background: rgba(255,255,255,0.4);
    }
    .my_post_copyright p{margin:0;}
    .my_post_copyright span {
    display: inline-block;
    width: 5.2em;
    color: #b5b5b5;
    font-weight: bold;
    }
    .my_post_copyright .raw {
    margin-left: 1em;
    width: 5em;
    }
    .my_post_copyright a {
    color: #808080;
    border-bottom:0;
    }
    .my_post_copyright a:hover {
    color: #a3d2a3;
    text-decoration: underline;
    }
    .my_post_copyright:hover .fa-clipboard {
    color: #000;
    }
    .my_post_copyright .post-url:hover {
    font-weight: normal;
    }
    .my_post_copyright .copy-path {
    margin-left: 1em;
    width: 1em;
    +mobile(){display:none;}
    }
    .my_post_copyright .copy-path:hover {
    color: #808080;
    cursor: pointer;
    }

修改`next/layout/_macro/post.swig`，在代码

    <div>
        {% if not is_index %}
            {% include 'wechat-subscriber.swig' %}
        {% endif %}
    </div>

之前添加增加如下代码：

    <div>
        {% if not is_index %}
            {% include 'my-copyright.swig' %}
        {% endif %}
    </div>

修改`next/source/css/_common/components/post/post.styl`文件，在最后一行增加代码：

    @import "my-post-copyright"

保存重新生成。如果要在该博文下面增加版权信息的显示，需要在 Markdown 中增加copyright: true的设置，类似：

    ---
    title: 前端小项目：使用canvas绘画哆啦A梦
    date: 2017-05-22 22:53:53
    tags: canvas
    categories: 前端
    copyright: true
    ---

如果你觉得每次都要输入`copyright: true`很麻烦的话,那么在`/scaffolds/post.md`文件中添加：

    ---
    title: {{ title }}
    date: {{ date }}
    tags:
    categories:
    copyright: true
    ---

***
<h4 id="flow">跟帖功能</h4>

更新next主题，因为最新版本的主题已经支持这种评论。直接在主题配置文件`_config.yml` 文件中添加如下配置:(来必力中的`data-uid`)

    # Support for LiveRe comments system.
    # You can get your uid from https://livere.com/insight/myCode (General web site)
    livere_uid: your uid

在 `layout/_scripts/third-party/comments/` 目录中添加 `livere.swig`，文件内容如下：

```
{% if not (theme.duoshuo and theme.duoshuo.shortname) and not theme.duoshuo_shortname and not theme.disqus_shortname and not theme.hypercomments_id and not theme.gentie_productKey %}
  {% if theme.livere_uid %}
    <script type="text/javascript">
      (function(d, s) {
        var j, e = d.getElementsByTagName(s)[0];
        if (typeof LivereTower === 'function') { return; }
        j = d.createElement(s);
        j.src = 'https://cdn-city.livere.com/js/embed.dist.js';
        j.async = true;
        e.parentNode.insertBefore(j, e);
      })(document, 'script');
    </script>
  {% endif %}
{% endif %}
```

然后在 `layout/_scripts/third-party/comments.swig`文件中追加：

    {% include './comments/livere.swig' %}

最后，在 `layout/_partials/comments.swig` 文件中条件最后追加 LiveRe 插件是否引用的判断逻辑：

```
{% elseif theme.livere_uid %}
      <div id="lv-container" data-id="city" data-uid="{{ theme.livere_uid }}"></div>
{% endif %}
```

***

<h4 id="secret">文章加密</h4>

打开`themes->next->layout->_partials->head.swig`文件,在以下位置插入代码：

    <script>
        (function(){
            if('{{ page.password }}'){
                if (prompt('请输入文章密码') !== '{{ page.password }}'){
                    alert('密码错误！');
                    history.back();
                }
            }
        })();
    </script>

在文章编辑的`head`中添加：

    password: password

***
<h4 id="share">文章分享</h4>

在`主题配置`文件中修改配置：

    jiathis: true

也可以通过修改`themes/next/layout/_partials/share/jiathis.swig`进行自定义。

***
<h4 id="top">博文置顶</h4>

修改 `hero-generator-index` 插件，把文件：`node_modules/hexo-generator-index/lib/generator.js` 内的代码替换为：

    'use strict';
    var pagination = require('hexo-pagination');
    module.exports = function(locals){
    var config = this.config;
    var posts = locals.posts;
        posts.data = posts.data.sort(function(a, b) {
            if(a.top && b.top) { // 两篇文章top都有定义
                if(a.top == b.top) return b.date - a.date; // 若top值一样则按照文章日期降序排
                else return b.top - a.top; // 否则按照top值降序排
            }
            else if(a.top && !b.top) { // 以下是只有一篇文章top有定义，那么将有top的排在前面（这里用异或操作居然不行233）
                return -1;
            }
            else if(!a.top && b.top) {
                return 1;
            }
            else return b.date - a.date; // 都没定义按照文章日期降序排
        });
    var paginationDir = config.pagination_dir || 'page';
    return pagination('', posts, {
        perPage: config.index_generator.per_page,
        layout: ['index', 'archive'],
        format: paginationDir + '/%d/',
        data: {
        __index: true
        }
    });
    };

在文章中添加`top`值，数值越大文章越靠前。

    top: 100

***
<h4 id="mouse">鼠标样式</h4>

打开`themes/next/source/css/_custom/custom.styl`,在里面写下如下代码

    // 鼠标样式
    * {
        cursor: url("http://om8u46rmb.bkt.clouddn.com/sword2.ico"),auto!important
    }
    :active {
        cursor: url("http://om8u46rmb.bkt.clouddn.com/sword1.ico"),auto!important
    }

其中 url 里面必须是 ico 图片。

***
<h4 id="animal">添加宠物</h4>

在根目录下，安装插件:

    npm install -save hexo-helper-live2d

然后打开`Hexo/blog/themes/next/layout`的`_layout.swig`,将下面代码放到`</body>`之前：

    {{ live2d() }}

然后在在 `hexo` 的 `_config.yml`中添加参数：

    live2d:
        model: wanko
        bottom: -30
        mobileShow: false #手机上不显示

重新生成文件：

    hexo clean
    hexo g
    hexo d

其它参数：

- model 模型名称 默认值: z16

        Gantzert_Felixander     Epsilon2.1  haru    miku     ni-j
        nico    nito   nipsilon    nietzsche    shizuku     tsumiki
        wanko   z16     hibiki  koharu  haruto  Unitychan   tororo  hijik

- width 宽度 默认值: 150
- height 高度 默认值：300
- className `<canvas>`元素的类名 默认值： live2d
- id `<canvas>` 元素的id 默认值： live2dcanvas
- bottom `<canvas>` 元素的底部偏移 默认值： -20 如果嫌模型位置不正确 可以调整这个参数

若出现部署错误，首先删除hexo 下面的.deploy_git文件夹，然后运行

    git config --global core.autocrlf false

***
<h4 id="chart">DaoVoice在线联系</h4>

首先在 [daovoice](https://account.daocloud.io/signin) 注册账号,注册完成后会得到一个 `app_id` ,调用`JavaScript`，获取app_id。

    daovoice('init',{
        app_id: ''
    });
    daovoice('update');

打开`/themes/next/layout/_partials/head.swig`,写下如下代码：

```
{% if theme.daovoice %}
  <script>
  (function(i,s,o,g,r,a,m){i["DaoVoiceObject"]=r;i[r]=i[r]||function(){(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;a.charset="utf-8";m.parentNode.insertBefore(a,m)})(window,document,"script",('https:' == document.location.protocol ? 'https:' : 'http:') + "//widget.daovoice.io/widget/0f81ff2f.js","daovoice")
  daovoice('init', {
      app_id: "{{theme.daovoice_app_id}}"
    });
  daovoice('update');
  </script>
{% endif %}
```

打开`主题配置`文件，在最后写下如下代码：

    # Online contact
    daovoice: true

    daovoice_app_id: 这里填你的刚才获得的 app_id

重新 `hexo g` ，`hexo s`，可以在DaoVoice 控制台上的聊天设置里设置聊天窗口样式。

***
<h4 id="boom">点击爆炸效果</h4>

首先在`themes/next/source/js/src`里面建一个叫`fireworks.js`的文件，代码如下：

```
"use strict";function updateCoords(e){pointerX=(e.clientX||e.touches[0].clientX)-canvasEl.getBoundingClientRect().left,pointerY=e.clientY||e.touches[0].clientY-canvasEl.getBoundingClientRect().top}function setParticuleDirection(e){var t=anime.random(0,360)*Math.PI/180,a=anime.random(50,180),n=[-1,1][anime.random(0,1)]*a;return{x:e.x+n*Math.cos(t),y:e.y+n*Math.sin(t)}}function createParticule(e,t){var a={};return a.x=e,a.y=t,a.color=colors[anime.random(0,colors.length-1)],a.radius=anime.random(16,32),a.endPos=setParticuleDirection(a),a.draw=function(){ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.fillStyle=a.color,ctx.fill()},a}function createCircle(e,t){var a={};return a.x=e,a.y=t,a.color="#F00",a.radius=0.1,a.alpha=0.5,a.lineWidth=6,a.draw=function(){ctx.globalAlpha=a.alpha,ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.lineWidth=a.lineWidth,ctx.strokeStyle=a.color,ctx.stroke(),ctx.globalAlpha=1},a}function renderParticule(e){for(var t=0;t<e.animatables.length;t++){e.animatables[t].target.draw()}}function animateParticules(e,t){for(var a=createCircle(e,t),n=[],i=0;i<numberOfParticules;i++){n.push(createParticule(e,t))}anime.timeline().add({targets:n,x:function(e){return e.endPos.x},y:function(e){return e.endPos.y},radius:0.1,duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule}).add({targets:a,radius:anime.random(80,160),lineWidth:0,alpha:{value:0,easing:"linear",duration:anime.random(600,800)},duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule,offset:0})}function debounce(e,t){var a;return function(){var n=this,i=arguments;clearTimeout(a),a=setTimeout(function(){e.apply(n,i)},t)}}var canvasEl=document.querySelector(".fireworks");if(canvasEl){var ctx=canvasEl.getContext("2d"),numberOfParticules=30,pointerX=0,pointerY=0,tap="mousedown",colors=["#FF1461","#18FF92","#5A87FF","#FBF38C"],setCanvasSize=debounce(function(){canvasEl.width=2*window.innerWidth,canvasEl.height=2*window.innerHeight,canvasEl.style.width=window.innerWidth+"px",canvasEl.style.height=window.innerHeight+"px",canvasEl.getContext("2d").scale(2,2)},500),render=anime({duration:1/0,update:function(){ctx.clearRect(0,0,canvasEl.width,canvasEl.height)}});document.addEventListener(tap,function(e){"sidebar"!==e.target.id&&"toggle-sidebar"!==e.target.id&&"A"!==e.target.nodeName&&"IMG"!==e.target.nodeName&&(render.play(),updateCoords(e),animateParticules(pointerX,pointerY))},!1),setCanvasSize(),window.addEventListener("resize",setCanvasSize,!1)}"use strict";function updateCoords(e){pointerX=(e.clientX||e.touches[0].clientX)-canvasEl.getBoundingClientRect().left,pointerY=e.clientY||e.touches[0].clientY-canvasEl.getBoundingClientRect().top}function setParticuleDirection(e){var t=anime.random(0,360)*Math.PI/180,a=anime.random(50,180),n=[-1,1][anime.random(0,1)]*a;return{x:e.x+n*Math.cos(t),y:e.y+n*Math.sin(t)}}function createParticule(e,t){var a={};return a.x=e,a.y=t,a.color=colors[anime.random(0,colors.length-1)],a.radius=anime.random(16,32),a.endPos=setParticuleDirection(a),a.draw=function(){ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.fillStyle=a.color,ctx.fill()},a}function createCircle(e,t){var a={};return a.x=e,a.y=t,a.color="#F00",a.radius=0.1,a.alpha=0.5,a.lineWidth=6,a.draw=function(){ctx.globalAlpha=a.alpha,ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.lineWidth=a.lineWidth,ctx.strokeStyle=a.color,ctx.stroke(),ctx.globalAlpha=1},a}function renderParticule(e){for(var t=0;t<e.animatables.length;t++){e.animatables[t].target.draw()}}function animateParticules(e,t){for(var a=createCircle(e,t),n=[],i=0;i<numberOfParticules;i++){n.push(createParticule(e,t))}anime.timeline().add({targets:n,x:function(e){return e.endPos.x},y:function(e){return e.endPos.y},radius:0.1,duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule}).add({targets:a,radius:anime.random(80,160),lineWidth:0,alpha:{value:0,easing:"linear",duration:anime.random(600,800)},duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule,offset:0})}function debounce(e,t){var a;return function(){var n=this,i=arguments;clearTimeout(a),a=setTimeout(function(){e.apply(n,i)},t)}}var canvasEl=document.querySelector(".fireworks");if(canvasEl){var ctx=canvasEl.getContext("2d"),numberOfParticules=30,pointerX=0,pointerY=0,tap="mousedown",colors=["#FF1461","#18FF92","#5A87FF","#FBF38C"],setCanvasSize=debounce(function(){canvasEl.width=2*window.innerWidth,canvasEl.height=2*window.innerHeight,canvasEl.style.width=window.innerWidth+"px",canvasEl.style.height=window.innerHeight+"px",canvasEl.getContext("2d").scale(2,2)},500),render=anime({duration:1/0,update:function(){ctx.clearRect(0,0,canvasEl.width,canvasEl.height)}});document.addEventListener(tap,function(e){"sidebar"!==e.target.id&&"toggle-sidebar"!==e.target.id&&"A"!==e.target.nodeName&&"IMG"!==e.target.nodeName&&(render.play(),updateCoords(e),animateParticules(pointerX,pointerY))},!1),setCanvasSize(),window.addEventListener("resize",setCanvasSize,!1)};
打开themes/next/layout/_layout.swig,在</body>上面写下如下代码：

{% if theme.fireworks %}
   <canvas class="fireworks" style="position: fixed;left: 0;top: 0;z-index: 1; pointer-events: none;" ></canvas>
   <script type="text/javascript" src="//cdn.bootcss.com/animejs/2.2.0/anime.min.js"></script>
   <script type="text/javascript" src="/js/src/fireworks.js"></script>
{% endif %}
```

打开`主题配置`文件，加入：

    # Fireworks
    fireworks: true

***
<h3 id="problems">常见问题</h3>

<h4 id="search">localsearch失效</h4>

hexo搜索失效，‘network’选项卡中可以看到正常请求search.xml文件且返回了200。错误如下：

    error on line 6 at column 35: Input is not proper UTF-8, indicate encoding ! Bytes: 0x08 0x79 0x6F 0x6C

主要原因是在博文中有非法字符，解决办法：在博文中搜索`\x08`,替换为无

***
<h3 id="relate">参考内容</h3>

1、https://www.jianshu.com/p/e8e5addbbcfd
2、https://www.jianshu.com/p/f054333ac9e6
3、https://www.jianshu.com/p/a2fe56d11c4f

***