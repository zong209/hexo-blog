---
title: Markdown语法
date: 2018-05-01 16:38:16
categories: 教程
tags: 
- Markown
- 语法
top: 10
---

Markdown是一种轻量级标记语言，它用简洁的语法代替排版，使我们专心于码字。它的目标是实现易读易写，成为一种适用于网络的书写语言。同时，Markdown支持嵌入html标签。
<!-- <font size="2">
<u>注：Markdown使用#、+、*等符号来标记， 符号后面必须跟上<b>至少1个</b>空格才有效！</u>
</font> -->

<h3 id="overview">Markdown特点</h3>

<h4 id="html">兼容 HTML</h4>

Markdown 的格式语法只涵盖纯文本可以涵盖的范围，不在 Markdown 涵盖范围之内的标签，都可以直接在文件里面用 HTML 撰写。不需要额外标注这是 HTML 或是 Markdown；只要直接加标签即可。

在 Markdown 文件里加上一段 HTML 表格：

    This is a regular paragraph.

    <table>
        <tr>
            <td>Foo</td>
        </tr>
    </table>

    This is another regular paragraph.
注意：

* 区块元素──比如 `<div>`、`<table>`、`<pre>`、`<p>` 等标签，必须在前后加上空行与其它内容区隔开，而且这些（元素）的开始与结尾标签，不可以用制表符或空格来缩进。
* Markdown 语法在 HTML 区块标签中将不会被进行处理，而Markdown 语法在 HTML 区段标签间是有效的。
* HTML 的区段（行内）标签如 `<span>`、`<cite>`、`<del>` 则不受限制，可以在 Markdown 的段落、列表或是标题里任意使用。依照个人习惯，甚至可以不用Markdown 格式，而直接采用 HTML 标签来格式化。

*****

<h4 id="autoescape">特殊字符自动转换</h4>

在 HTML 文件中，有两个字符需要特殊处理： `<` 和 `&` 。 `<` 符号用于起始标签，`&` 符号则用于标记 HTML 实体，如果你只是想要显示这些字符的原型，你必须要使用实体的形式，像是 `&lt;` 和 `&amp;`。

`&` 符号其实很容易让写作网络文件的人感到困扰，如果你要打「AT&T」 ，你必须要写成「`AT&amp;T`」 ，而且网址中的 `&` 字符也要转换。如果你要链接到：

    http://images.google.com/images?num=30&q=larry+bird

你必须要把网址写成：

    http://images.google.com/images?num=30&amp;q=larry+bird

才能放到链接标签的 `href` 属性里。不用说也知道这很容易忽略，这也可能是 HTML 标准检查所检查到的错误中，数量最多的。

Markdown 可以自动处理这些符号，如果你使用的 `&` 字符是 HTML 字符实体的一部分，它会保留原状，否则它会被转换成 `&amp`;。
所以你如果要在文件中插入一个著作权的符号，你可以这样写：

    &copy;

Markdown 将不会对这段文字做修改，但是如果你这样写：

    AT&T

Markdown 就会将它转为：

    AT&amp;T

类似的状况也会发生在 `<` 符号上，因为 Markdown 支持 [兼容 HTML](#html) ，如果你是使用 `<` 符号作为 HTML 标签使用，那 Markdown 也不会对它做任何转换，但是如果你是写：

    4 < 5

Markdown 将会把它转换为：

    4 &lt; 5

不过需要注意的是，code 范围内，不论是行内还是区块， `<` 和 `&` 两个符号都**一定**会被转换成 HTML 实体，这项特性让你可以很容易地用 Markdown 写 HTML code （和 HTML 相对而言， HTML 语法中，你要把所有的 `<` 和 `&` 都转换为 HTML 实体，才能在 HTML 文件里面写出 HTML code。）

*****
<h3 id="block">区块元素</h3>

<h4 id="p">段落和换行</h4>

一个 Markdown 段落是由一个或多个连续的文本行组成，它的前后要有一个以上的空行（空行的定义是显示上看起来像是空的，便会被视为空行。比方说，若某一行只包含空格和制表符，则该行也会被视为空行）。普通段落不该用空格或制表符来缩进。

「由一个或多个连续的文本行组成」这句话其实暗示了 Markdown 允许段落内的强迫换行（插入换行符），这个特性和其他大部分的 text-to-HTML 格式不一样（包括 Movable Type 的「Convert Line Breaks」选项），其它的格式会把每个换行符都转成 `<br />` 标签。

如果你**真的**想要插入 `<br />` 标签的话，在行尾加上两个以上的格然后回车。

是的，这确实需要花比较多功夫来插入 `<br />` ，但是「每个换行都转换为 `<br />`」的方法在 Markdown 中并不适合， Markdown 中 email 式的 [区块引用][bq] 和多段落的 [列表][l] 在使用换行来排版的时候，不但更好用，还更好阅读。

  [bq]: #blockquote
  [l]:  #list

*****

<h4 id="header">标题</h4>

Markdown 支持两种标题的语法，类 [Setext] [1] 和类 [atx] [2] 形式。

类 Setext 形式是用底线的形式，利用 `=` （最高阶标题）和 `-` （第二阶标题），例如：

    This is an H1
    =============

    This is an H2
    -------------

任何数量的 `=` 和 `-` 都可以有效果。

类 Atx 形式则是在行首插入 1 到 6 个 `#` ，对应到标题 1 到 6 阶，例如：

    # This is an H1

    ## This is an H2

    ###### This is an H6

你可以选择性地「闭合」类 atx 样式的标题，这纯粹只是美观用的，若是觉得这样看起来比较舒适，你就可以在行尾加上 `#`，而行尾的 `#` 数量也不用和开头一样（行首的井字数量决定标题的阶数）：

    # This is an H1 #

    ## This is an H2 ##

    ### This is an H3 ######

*****

<h4 id="blockquote">Blockquotes</h4>

Markdown 标记区块引用是使用类似 email 中用 `>` 的引用方式。如果你还熟悉在 email 信件中的引言部分，你就知道怎么在 Markdown 文件中建立一个区块引用，那会看起来像是你自己先断好行，然后在每行的最前面加上 `>` ：

    > This is a blockquote with two paragraphs. Lorem ipsum dolor sit amet,
    > consectetuer adipiscing elit. Aliquam hendrerit mi posuere lectus.
    > Vestibulum enim wisi, viverra nec, fringilla in, laoreet vitae, risus.
    >
    > Donec sit amet nisl. Aliquam semper ipsum sit amet velit. Suspendisse
    > id sem consectetuer libero luctus adipiscing.

Markdown 也允许你只在整个段落的第一行最前面加上 `>` ：

    > This is a blockquote with two paragraphs. Lorem ipsum dolor sit amet,
    consectetuer adipiscing elit. Aliquam hendrerit mi posuere lectus.
    Vestibulum enim wisi, viverra nec, fringilla in, laoreet vitae, risus.

    > Donec sit amet nisl. Aliquam semper ipsum sit amet velit. Suspendisse
    id sem consectetuer libero luctus adipiscing.

区块引用可以有嵌套（例如：引用内的引用），只要根据层数加上不同数量的 `>` ：

    > This is the first level of quoting.
    >
    > > This is nested blockquote.
    >
    > Back to the first level.

引用的区块内也可以使用其他的 Markdown 语法，包括标题、列表、代码区块等：

    > ## This is a header.
    >
    > 1.   This is the first list item.
    > 2.   This is the second list item.
    >
    > Here's some example code:
    >
    >     return shell_exec("echo $input | $markdown_script");

任何标准的文本编辑器都能简单地建立 email 样式的引用，例如 BBEdit ，你可以选取文字后然后从选单中选择*增加引用阶层*。

*****

<h4 id="list">列表</h4>

Markdown 支持有序列表和无序列表。

无序清单使用星号、加号或是减号作为列表标记：

    *   Red
    *   Green
    *   Blue

等同于：

    +   Red
    +   Green
    +   Blue

也等同于：

    -   Red
    -   Green
    -   Blue

有序列表则使用数字接着一个英文句点：

    1.  Bird
    2.  McHale
    3.  Parish

很重要的一点是，你在列表标记上使用的数字并不会影响输出的 HTML 结果，上面的列表所产生的 HTML 标记为：

    <ol>
    <li>Bird</li>
    <li>McHale</li>
    <li>Parish</li>
    </ol>

如果你的列表标记写成：

    1.  Bird
    1.  McHale
    1.  Parish

或甚至是：

    3. Bird
    1. McHale
    8. Parish

你都会得到完全相同的 HTML 输出。重点在于，你可以让 Markdown 文件的列表数字和输出的结果相同，或是你懒一点，你可以完全不用在意数字的正确性。

如果你使用懒惰的写法，建议第一个项目最好还是从 1. 开始，因为 Markdown 未来可能会支持有序列表的 start 属性。

列表项目标记通常是放在最左边，但是其实也可以缩进，**最多 3 个空格**，项目标记后面则一定要接着至少一个空格或制表符。

要让列表看起来更漂亮，你可以把内容用固定的缩进整理好：

    *   Lorem ipsum dolor sit amet, consectetuer adipiscing elit.
        Aliquam hendrerit mi posuere lectus. Vestibulum enim wisi,
        viverra nec, fringilla in, laoreet vitae, risus.
    *   Donec sit amet nisl. Aliquam semper ipsum sit amet velit.
        Suspendisse id sem consectetuer libero luctus adipiscing.

但是如果你很懒，那也行：

    *   Lorem ipsum dolor sit amet, consectetuer adipiscing elit.
    Aliquam hendrerit mi posuere lectus. Vestibulum enim wisi,
    viverra nec, fringilla in, laoreet vitae, risus.
    *   Donec sit amet nisl. Aliquam semper ipsum sit amet velit.
    Suspendisse id sem consectetuer libero luctus adipiscing.

如果列表项目间用空行分开， 在输出 HTML 时 Markdown 会把项目的内容在输出时用 `<p>`
标签包起来，举例来说：

    *   Bird
    *   Magic

会被转换为：

    <ul>
    <li>Bird</li>
    <li>Magic</li>
    </ul>

但是这个：

    *   Bird

    *   Magic

会被转换为：

    <ul>
    <li><p>Bird</p></li>
    <li><p>Magic</p></li>
    </ul>

列表项目可以包含多个段落，每个项目下的段落都必须缩进 4 个空格或是 1 个制表符：

    1.  This is a list item with two paragraphs. Lorem ipsum dolor
        sit amet, consectetuer adipiscing elit. Aliquam hendrerit
        mi posuere lectus.

        Vestibulum enim wisi, viverra nec, fringilla in, laoreet
        vitae, risus. Donec sit amet nisl. Aliquam semper ipsum
        sit amet velit.

    2.  Suspendisse id sem consectetuer libero luctus adipiscing.

如果你每行都有缩进，看起来会看好很多，当然，Markdown 也允许：

    *   This is a list item with two paragraphs.

        This is the second paragraph in the list item. You're
    only required to indent the first line. Lorem ipsum dolor
    sit amet, consectetuer adipiscing elit.

    *   Another item in the same list.

如果要在列表项目内放进引用，那 `>` 就需要缩进：

    *   A list item with a blockquote:

        > This is a blockquote
        > inside a list item.

如果要放代码区块的话，该区块就需要缩排**两次**，也就是 8 个空格或是 2 个制表符：

    *   A list item with a code block:

            <code goes here>

当然，项目列表很可能会不小心产生，像是下面这样的写法：

    1986. What a great season.

换句话说，也就是在行首出现**数字-句点-空白**，要避免这样的状况，你可以在句点前面加上反斜杠。

    1986\. What a great season.

*****

<h4 id="precode">程序代码区块</h4>

和程序相关的写作或是标签语言原始码通常会有已经排版好的代码区块，通常这些区块我们并不希望它以一般段落文件的方式去排版，而是照原来的样子显示，Markdown 会用 `<pre>` 和 `<code>` 标签来把代码区块包起来。

要在 Markdown 中建立代码区块很简单，只要简单地缩进 4 个空格或是 1 个制表符就可以，例如，下面的输入：

    This is a normal paragraph:

        This is a code block.

Markdown 会转换成：

    <p>This is a normal paragraph:</p>

    <pre><code>This is a code block.
    </code></pre>

这个每行一阶的缩进（4 个空格或是 1 个制表符），都会被移除，例如：

    Here is an example of AppleScript:

        tell application "Foo"
            beep
        end tell

会被转换为：

    <p>Here is an example of AppleScript:</p>

    <pre><code>tell application "Foo"
        beep
    end tell
    </code></pre>

一个代码区块会一直持续到没有缩进的那一行（或是文件结尾）。

在代码区块里面， `&` 、 `<` 和 `>` 会自动转成 HTML 实体，这样的方式让你非常容易使用 Markdown 插入范例用的 HTML 原始码，只需要复制贴上，再加上缩进就可以了，剩下的 Markdown 都会帮你处理，例如：

    <div class="footer">
        &copy; 2004 Foo Corporation
    </div>

会被转换为：

    <pre><code>&lt;div class="footer"&gt;
        &amp;copy; 2004 Foo Corporation
    &lt;/div&gt;
    </code></pre>

代码区块中，一般的 Markdown 语法不会被转换，像是星号便只是星号，这表示你可以很容易地以 Markdown 语法撰写 Markdown 语法相关的文件。

*****

<h4 id="hr">分隔线</h4>

你可以在一行中用三个或以上的星号、减号来建立一个分隔线，行内不能有其他东西。你也可以在星号中间插入空格。下面每种写法都可以建立分隔线：

    * * *

    ***

    *****

    - - -

    ---------------------------------------

*****

<h3 id="span">区段元素</h3>

<h4 id="link">链接</h4>

Markdown 支持两种形式的链接语法： **行内式**和**参考式**两种形式。

不管是哪一种，链接的文字都是用 [方括号] 来标记。

要建立一个**行内式**的链接，只要在方括号后面马上接着圆括号并插入网址链接即可，如果你还想要加上链接的 title 文字，只要在网址后面，用双引号把 title 文字包起来即可，例如：

    This is [an example](http://example.com/ "Title") inline link.

    [This link](http://example.net/) has no title attribute.

会产生：

    <p>This is <a href="http://example.com/" title="Title">
    an example</a> inline link.</p>

    <p><a href="http://example.net/">This link</a> has no
    title attribute.</p>

如果你是要链接到同样主机的资源，你可以使用相对路径：

    See my [About](/about/) page for details.

**参考式**的链接是在链接文字的括号后面再接上另一个方括号，而在第二个方括号里面要填入用以辨识链接的标记：

    This is [an example][id] reference-style link.

你也可以选择性地在两个方括号中间加上空格：

    This is [an example] [id] reference-style link.

接着，在文件的任意处，你可以把这个标签的链接内容定义出来：

    [id]: http://example.com/  "Optional Title Here"

链接内容定义的形式为：

* 方括号（前面可以选择性地加上至多三个空格来缩进），里面输入链接文字
* 接着一个冒号
* 接着一个以上的空格或制表符
* 接着链接的网址
* 选择性地接着 title 内容，可以用单引号、双引号或是圆括号包着

下面这三种连结的定义都是相同：

    [foo]: http://example.com/  "Optional Title Here"
    [foo]: http://example.com/  'Optional Title Here'
    [foo]: http://example.com/  (Optional Title Here)

**请注意：**有一个已知的问题是 Markdown.pl 1.0.1 会忽略单引号包起来的链接 title。

链接网址也可以用方括号包起来：

    [id]: <http://example.com/>  "Optional Title Here"

你也可以把 title 属性放到下一行，也可以加一些缩进，网址太长的话，这样会比较好看：

    [id]: http://example.com/longish/path/to/resource/here
        "Optional Title Here"

网址定义只有在产生链接的时候用到，并不会直接出现在文件之中。

链接辨识标记可以有字母、数字、空白和标点符号，但是并**不区分**大小写，因此下面两个链接是一样的：

    [link text][a]
    [link text][A]

**隐式链接标记**功能让你可以省略指定链接标记，这种情形下，链接标记和链接文字会视为相同，要用隐式链接标记只要在链接文字后面加上一个空的方括号，如果你要让 "Google" 链接到 google.com，你可以简化成：

    [Google][]

然后定义链接内容：

    [Google]: http://google.com/

由于链接文字可能包含空白，所以这种简化的标记内也可以包含多个单词：

    Visit [Daring Fireball][] for more information.

然后接着定义链接：

    [Daring Fireball]: http://daringfireball.net/

链接的定义可以放在文件中的任何一个地方，我比较偏好直接放在链接出现段落的后面，你也可以把它放在文件最后面，就像是注解一样。

下面是一个参考式链接的范例：

    I get 10 times more traffic from [Google] [1] than from
    [Yahoo] [2] or [MSN] [3].

      [1]: http://google.com/        "Google"
      [2]: http://search.yahoo.com/  "Yahoo Search"
      [3]: http://search.msn.com/    "MSN Search"

如果改成用链接名称的方式写：

    I get 10 times more traffic from [Google][] than from
    [Yahoo][] or [MSN][].

      [google]: http://google.com/        "Google"
      [yahoo]:  http://search.yahoo.com/  "Yahoo Search"
      [msn]:    http://search.msn.com/    "MSN Search"

上面两种写法都会产生下面的 HTML。

    <p>I get 10 times more traffic from <a href="http://google.com/"
    title="Google">Google</a> than from
    <a href="http://search.yahoo.com/" title="Yahoo Search">Yahoo</a>
    or <a href="http://search.msn.com/" title="MSN Search">MSN</a>.</p>

下面是用行内形式写的同样一段内容的 Markdown 文件，提供作为比较之用：

    I get 10 times more traffic from [Google](http://google.com/ "Google")
    than from [Yahoo](http://search.yahoo.com/ "Yahoo Search") or
    [MSN](http://search.msn.com/ "MSN Search").

参考式的链接其实重点不在于它比较好写，而是它比较好读，比较一下上面的范例，使用参考式的文章本身只有 81 个字符，但是用行内形式的连结却会增加到 176 个字符，如果是用纯 HTML 格式来写，会有 234 个字符，在 HTML 格式中，标签比文字还要多。

使用 Markdown 的参考式链接，可以让文件更像是浏览器最后产生的结果，让你可以把一些标记相关的信息移到段落文字之外，你就可以增加连结而不让文章的阅读感觉被打断。

*****

<h4 id="achor">锚点</h4>

网页中，锚点其实就是页内超链接，也就是链接本文档内部的某些元素，实现当前页面中的跳转。比如我这里写下一个锚点，点击回到目录，就能跳转到目录。 在目录中点击这一节，就能跳过来。还有下一节的注脚。这些根本上都是用锚点来实现的。

注意：

* Markdown Extra 只支持在标题后插入锚点，其它地方无效。
* Leanote 编辑器右侧显示效果区域暂时不支持锚点跳转，所以点来点去发现没有跳转不必惊慌，但是你发布成笔记或博文后是支持跳转的。

例如：

    ## 0. 目录{#index}

    跳转到[目录](#index)

*****

<h4 id="foot">注脚</h4>

在需要添加注脚的文字后加上脚注名字[^注脚名字],称为加注。 然后在文本的任意位置(一般在最后)添加脚注，脚注前必须有对应的脚注名字。
注意：经测试注脚与注脚之间必须空一行，不然会失效。成功后会发现，即使你没有把注脚写在文末，经Markdown转换后，也会自动归类到文章的最后。

    使用 Markdown[^1]可以效率的书写文档, 直接转换成 HTML[^2], 你可以使用 Leanote[^Le] 编辑器进行书写。
    [^1]:Markdown是一种纯文本标记语言
    [^2]:HyperText Markup Language 超文本标记语言
    [^Le]:开源笔记平台，支持Markdown和笔记直接发为博文

*****

<h4 id="em">强调</h4>

Markdown 使用星号（`*`）和底线（`_`）作为标记强调字词的符号，被 `*` 或 `_` 包围的字词会被转成用 `<em>` 标签包围，用两个 `*` 或 `_` 包起来的话，则会被转成 `<strong>`，例如：

    *single asterisks*

    _single underscores_

    **double asterisks**

    __double underscores__

会转成：

    <em>single asterisks</em>

    <em>single underscores</em>

    <strong>double asterisks</strong>

    <strong>double underscores</strong>

你可以随便用你喜欢的样式，唯一的限制是，你用什么符号开启标签，就要用什么符号结束。

强调也可以直接插在文字中间：

    un*frigging*believable

但是**如果你的 `*` 和 `_` 两边都有空白的话，它们就只会被当成普通的符号**。

如果要在文字前后直接插入普通的星号或底线，你可以用反斜杠：

    \*this text is surrounded by literal asterisks\*

*****

<h4 id="code">代码</h4>

如果要标记一小段行内代码，你可以用反引号把它包起来（`` ` ``），例如：

    Use the `printf()` function.

会产生：

    <p>Use the <code>printf()</code> function.</p>

如果要在代码区段内插入反引号，你可以用多个反引号来开启和结束代码区段：

    ``There is a literal backtick (`) here.``

这段语法会产生：

    <p><code>There is a literal backtick (`) here.</code></p>

代码区段的起始和结束端都可以放入一个空白，起始端后面一个，结束端前面一个，这样你就可以在区段的一开始就插入反引号：

    A single backtick in a code span: `` ` ``

    A backtick-delimited string in a code span: `` `foo` ``

会产生：

    <p>A single backtick in a code span: <code>`</code></p>

    <p>A backtick-delimited string in a code span: <code>`foo`</code></p>

在代码区段内，`&` 和方括号都会被转成 HTML 实体，这样会比较容易插入 HTML 原始码，Markdown 会把下面这段：

    Please don't use any `<blink>` tags.

转为：

    <p>Please don't use any <code>&lt;blink&gt;</code> tags.</p>

你也可以这样写：

    `&#8212;` is the decimal-encoded equivalent of `&mdash;`.

以产生：

    <p><code>&amp;#8212;</code> is the decimal-encoded
    equivalent of <code>&amp;mdash;</code>.</p>

*****

<h4 id="img">图片</h4>

很明显地，要在纯文本应用中设计一个 「自然」的语法来插入图片是有一定难度的。

Markdown 使用一种和链接很相似的语法来标记图片，同样也允许两种样式： **行内式**和**参考式**。

行内式图片的语法看起来像是：

    ![Alt text](/path/to/img.jpg)

    ![Alt text](/path/to/img.jpg "Optional title")

详细叙述如下：

* 一个惊叹号 `!`
* 接着一对方括号，里面放上图片的替代文字
* 接着一对普通括号，里面放上图片的网址，最后还可以用引号包住并加上选择性的 'title' 文字。

参考式的图片语法则长得像这样：

    ![Alt text][id]

「id」是图片参考的名称，图片参考的定义方式则和链接参考一样：

    [id]: url/to/image  "Optional title attribute"

到目前为止， Markdown 还没有办法指定图片的宽高，如果你需要的话，你可以使用普通的 `<img>` 标签。

*****

<h3 id="misc">其它</h3>

<h4 id="autolink">自动链接</h4>

Markdown 支持比较简短的自动连结形式来处理网址和电子邮件信箱，只要是用方括号包起来， Markdown 就会自动把它转成链接，链接的文字就和链接位置一样，例如：

    <http://example.com/>

Markdown 会转为：

    <a href="http://example.com/">http://example.com/</a>

邮址的自动链接也很类似，只是 Markdown 会先做一个编码转换的过程，把文字字符转成 16 进位码的 HTML 实体，这样的格式可以糊弄一些不好的邮址收集机器人，例如：

    <address@example.com>

Markdown 会转成：

    <a href="&#x6D;&#x61;i&#x6C;&#x74;&#x6F;:&#x61;&#x64;&#x64;&#x72;&#x65;
    &#115;&#115;&#64;&#101;&#120;&#x61;&#109;&#x70;&#x6C;e&#x2E;&#99;&#111;
    &#109;">&#x61;&#x64;&#x64;&#x72;&#x65;&#115;&#115;&#64;&#101;&#120;&#x61;
    &#109;&#x70;&#x6C;e&#x2E;&#99;&#111;&#109;</a>

在浏览器里面，这段字符串（其实是 `<a href="mailto:address@example.com">address@example.com</a>`）会变成一个可以点击的「address@example.com」链接。

（这种作法虽然可以糊弄不少的机器人，但并无法全部挡下来，不过这样也比什么都不做好些。无论如何，公开你的信箱终究会引来广告信件的。）

*****

<h4 id="backslash">反斜杠</h4>

Markdown 可以利用反斜杠来插入一些在语法中有其它意义的符号，例如：如果你想要用星号加在文字旁边的方式来做出强调效果（但不用 `<em>` 标签），你可以在星号的前面加上反斜杠：

    \*literal asterisks\*

Markdown 支持在下面这些符号前面加上反斜杠来帮助插入普通的符号：

    \   反斜杠
    `   反引号
    *   星号
    _   底线
    {}  大括号
    []  方括号
    ()  括号
    #   井字号
    +   加号
    -   减号
    .   英文句点
    !   惊叹号

*****

<h4 id="equation">公式</h4>

行内公式（$ 包裹），例如：

    质能守恒方程可以用一个很简洁的方程式 $E=mc^2$ 来表达。

整行公式（$$包裹），例如：

    $$\sum_{i=1}^n a_i=0$$

    $$f(x_1,x_x,\ldots,x_n) = x_1^2 + x_2^2 + \cdots + x_n^2 $$

    $$\sum^{j-1}_{k=0}{\widehat{\gamma}_{kj} z_k}$$

访问[MathJax][]参考更多使用方法。

[MathJax]: https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference

*****

<h4 id="graph">流程图</h4>

    (```)flow
    st=>start: Start|past:>http://www.google.com[blank]
    e=>end: End:>http://www.google.com
    op1=>operation: My Operation|past
    op2=>operation: Stuff|current
    sub1=>subroutine: My Subroutine|invalid
    cond=>condition: Yes
    or No?|approved:>http://www.google.com
    c2=>condition: Good idea|rejected
    io=>inputoutput: catch something...|request
    st->op1(right)->cond
    cond(yes, right)->c2
    cond(no)->sub1(left)->op1
    c2(yes)->io->e
    c2(no)->op2->e
    (```)

显示如下：

```flow
st=>start: Start|past:>http://www.google.com[blank]
e=>end: End:>http://www.google.com
op1=>operation: My Operation|past
op2=>operation: Stuff|current
sub1=>subroutine: My Subroutine|invalid
cond=>condition: Yes
or No?|approved:>http://www.google.com
c2=>condition: Good idea|rejected
io=>inputoutput: catch something...|request

st->op1(right)->cond
cond(yes, right)->c2
cond(no)->sub1(left)->op1
c2(yes)->io->e
c2(no)->op2->e
```

流程图分为两部分**定义参数**和**连接参数**

    tag=>type: content:>url         // 形参格式
    st=>start: 开始:>http://www.baidu.com[blank]  //实参格式

**注：`st=>start: 开始`的`:`后保持空格**

形参|实参|含义
:-:|:-:|:-:
tag|st|标签（可以自定义）
=>|=>|赋值
type|start|类型 (6种类型)
content|开始|描述内容 (可以自定义)
:>url|<http://www.baidu.com[blank]>|链接与跳转方式 兼容性很差

单元类型：

6种类型|含义
:-:|:-:|
start|启动
end|结束
operation|程序
subroutine|子程序
condition|条件
inputoutput|输出

连接示例：

    st->c1(yes,right)->c2(yes,right)->c3(yes,right)->io->e
    开始->判断条件1为no->判断条件2为no->判断条件3为no->输出->结束

形参|实参|含义
:-:|:-:|:-:
->|->|连接
condition|c1|条件
(布尔值,方向)|(yes,right)|如果满足向右连接，4种方向：right ，left，up ，down 默认为：down

`注：operation (程序); subroutine (子程序) ;condition (条件)，都可以在括号里加入连接方向`

    operation(right)
    subroutine(left)
    condition(yes,right)    // 只有条件 才能加布尔值

了解更多使用法，访问[流程图语法参考][]

[流程图语法参考]:http://flowchart.js.org/

*****

<h4 id="form">表格</h4>

基本语法：

* 第一行为表头，第二行分隔表头和主体部分，第三行开始每一行为一个表格行。
* 列于列之间用管道符|隔开。原生方式的表格每一行的两边也要有管道符。
* 第二行还可以为不同的列指定对齐方向。默认为左对齐，在-右边加上:就右对齐。

简单方式：

    学号|姓名|分数
    -|-|-
    小明|男|75
    小红|女|79
    小陆|男|92

原生方式：

    |学号|姓名|分数|
    |-|-|-|
    |小明|男|75|
    |小红|女|79|
    |小陆|男|92|

对齐方式：

    第一列 | 第二列 | 第三列
    -------: | :------: | :-------
    右对齐 | 居中 | 左对齐

*****

<h3 id="relate">参考内容</h3>

1、<https://github.com/othree/markdown-syntax-zhtw>
2、<https://blog.csdn.net/witnessai1/article/details/52551362>

*****