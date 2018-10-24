---
title: JS代码模块
date: 2018-05-02 15:30:03
categories: codeblock
tags:
- JavaScript
- Code
top: 100
---
常用的JavaScript功能模块。

<h4 id="deleteSpace">去除字符串中空格</h4>

1、去掉字符串前后所有空格

    function Trim(str)
    {
    return str.replace(/(^\s*)|(\s*$)/g, "");
    }

2、去掉字符串中所有空格(`is_global="g"`)

    function Trim(str, is_global) {
    var result;
    result = str.replace(/(^\s+)|(\s+$)/g, "");
    if (is_global.toLowerCase() == "g") {
        result = result.replace(/\s/g, "");
    }
    return result;
    }

3.日期格式化

        Date.prototype.format = function (formatStr) {
        var str = formatStr;
        var Week = ['日', '一', '二', '三', '四', '五', '六'];
        str = str.replace(/yyyy|YYYY/, this.getFullYear());
        str = str.replace(/MM/, (this.getMonth() + 1) > 9 ? (this.getMonth() + 1).toString() : '0' + (this.getMonth() + 1));
        str = str.replace(/dd|DD/, this.getDate() > 9 ? this.getDate().toString() : '0' + this.getDate());
        str = str.replace(/hh|HH/, this.getHours() > 9 ? this.getHours().toString() : '0' + this.getHours());
        str = str.replace(/mm/, this.getMinutes() > 9 ? this.getMinutes().toString() : '0' + this.getMinutes());
        str = str.replace(/ss|SS/, this.getSeconds() > 9 ? this.getSeconds().toString() : '0' + this.getSeconds());
        return str;
        } 

4.判断图片是否存在

    function isHasImg(pathImg){
        var ImgObj=new Image();
        ImgObj.src= pathImg;
        if(ImgObj.fileSize > 0 || (ImgObj.width > 0 && ImgObj.height > 0))
        {
        return true;
        } else {
        return false;
        }
    }       