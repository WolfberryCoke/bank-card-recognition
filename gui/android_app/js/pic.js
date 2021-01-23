function getImage() {
    var c = plus.camera.getCamera();
    c.captureImage(function(e) {
        plus.io.resolveLocalFileSystemURL(e, function(entry) {
            var s = entry.toLocalURL() + "?version=" + new Date().getTime();
            uploadHead(s); /*上传图片*/
        }, function(e) {
            console.log("读取拍照文件错误：" + e.message);
        });
    }, function(s) {
        console.log("error" + s);
    }, {
        filename: "_doc/head.png"
    })
}

//本地相册选择
function galleryImg() {
    plus.gallery.pick(function(a) {
        plus.io.resolveLocalFileSystemURL(a, function(entry) {
            plus.io.resolveLocalFileSystemURL("_doc/", function(root) {
                root.getFile("head.png", {}, function(file) {
                    //文件已存在
                    file.remove(function() {
                        console.log("file remove success");
                        entry.copyTo(root, 'head.png', function(e) {
                                var e = e.fullPath + "?version=" + new Date().getTime();
                                uploadHead(e); /*上传图片*/
                                //变更大图预览的src
                                //目前仅有一张图片，暂时如此处理，后续需要通过标准组件实现
                            },
                            function(e) {
                                console.log('copy image fail:' + e.message);
                            });
                    }, function() {
                        console.log("delete image fail:" + e.message);
                    });
                }, function() {
                    //文件不存在
                    entry.copyTo(root, 'head.png', function(e) {
                            var path = e.fullPath + "?version=" + new Date().getTime();
                            uploadHead(path); /*上传图片*/
                        },
                        function(e) {
                            console.log('copy image fail:' + e.message);
                        });
                });
            }, function(e) {
                console.log("get _www folder fail");
            })
        }, function(e) {
            console.log("读取拍照文件错误：" + e.message);
        });
    }, function(a) {}, {
        filter: "image"
    })
};