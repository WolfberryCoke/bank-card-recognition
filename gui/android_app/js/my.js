function upload(imgsrc){
	//服务端接口路径
    var server = "http://111.231.107.140:5000/upload";
    var wt=plus.nativeUI.showWaiting();
    var task=plus.uploader.createUpload(server,
        {method:"POST"},
        function(t,status){ //上传完成
            if(status==200){
                alert("上传成功："+t.responseText);
                wt.close(); //关闭等待提示按钮
            }else{
                alert("上传失败："+status);
                wt.close();//关闭等待提示按钮
            }
        }
    );  
    //添加其他参数
    task.addData("name","test");
    task.addFile(imgsrc,{key:"dddd"});
    task.start();
} 

//获取base64方法
function getBase64(url) { //传入图片路径
    function getBase64Image(img,width,height) {
        var canvas = document.createElement("canvas");
        canvas.width = width ? width : img.width;
        canvas.height = height ? height : img.height;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        dataBase64 = canvas.toDataURL("image/jpg");
        postImg(dataBase64.substr(22));//dataBase64上传到后台
    }
    var image = new Image();
    image.onload=function(){//onload事件不执行，后查是因为onloand事件是基于http协议的，file://。。.jpg路径没法执行，弃之 
        mui.toast("load2");
        getBase64Image(image);
    };
    image.src=url;
}