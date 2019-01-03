function postImg(){
        //执行post请求，识别图片
        jQuery("#billmodeltable").remove();//清空界面识别结果
       
         if(imgJson['num']==0)
         {   loadingGif('loadingGif');
             imgJson['num']=1;//防止重复提交
        jQuery.ajax({
            type: "post",
            url: 'ocr',
            data:JSON.stringify({"imgString":imgJson["imgString"]}),
          success:function(d){
              loadingGif('loadingGif');
              imgJson['num']=0;//防止重复提交
              res = JSON.parse(d);
              imgJson["result"] = res['res'];
              imgJson["timeTake"] = res['timeTake'];
              getChildDetail();
              W = imgJson["width"];
              H = imgJson["height"];
          }
        });}
        
         }


function loadingGif(loadingGif){
        //加载请求时旋转动态图片
        var imgId=document.getElementById(loadingGif);
        if(imgId.style.display=="block")
        {imgId.style.display="none";}
        else
        {imgId.style.display="block";}}


function resize_im(w,h, scale, max_scale){
    f=parseFloat(scale)/Math.min(h, w);
    if(f*Math.max(h, w)>max_scale){
            f=parseFloat.float(max_scale)/max(h, w);
    }
    newW = parseInt(w*f);
    newH    =parseInt(h*f);
    
    return [newW,newH]
}


function FunimgPreview(avatarSlect,avatarPreview,myCanvas) {
                //avatarSlect 上传文件控件
                //avatarPreview 预览图片控件
                jQuery("#"+avatarSlect).change(function () {
                var obj=jQuery("#"+avatarSlect)[0].files[0];
                
                var fr=new FileReader();
                fr.readAsDataURL(obj);
                fr.onload=function () {
                      jQuery("#"+avatarPreview).attr('src',this.result);
                      imgJson.imgString = this.result;
                      
                      var image = new Image();
                      image.onload=function(){
                                      var width = image.width;
                                      var height = image.height;
                                      newWH =resize_im(width,height, 800, 1600);
                                      newW = newWH[0];
                                      newH = newWH[1]; 
                                      imgJson.width = width;
                                      imgJson.height = height;
                                      jQuery("#"+avatarPreview).attr('width',newW);
                                      jQuery("#"+avatarPreview).attr('height',newH);
                                      jQuery("#"+'myCanvas').attr('width',newW);
                                      jQuery("#"+'myCanvas').attr('height',newH);
                          
                                      /*
                                      if(width>height){
                                      jQuery("#"+avatarPreview).attr('width',1600);
                                      jQuery("#"+avatarPreview).attr('height',800);
                                      jQuery("#"+'myCanvas').attr('width',1600);
                                      jQuery("#"+'myCanvas').attr('height',800);
                                      }
                                      else{
                                          jQuery("#"+avatarPreview).attr('width',600);
                                          jQuery("#"+avatarPreview).attr('height',1000);
                                          jQuery("#"+myCanvas).attr('width',600);
                                          jQuery("#"+myCanvas).attr('height',1000);
                                      }
                                      */
                                      };
                      image.src= this.result;
                      //box = {"xmin":0,"ymin":0,"xmax":jQuery("#"+'myCanvas').width(),"ymax":jQuery("#"+'myCanvas').height()};                         //createNewCanvas(this.result,'myCanvas',box);
                      
                  
                postImg();//提交POST请求
                };//fr.onload
                
                })//jQuery("#"+avatarSlect)
 }
    
function getChildDetail(){
  jQuery("#billmodeltable").remove();
  childResult = imgJson["result"];
  createTable(childResult,imgJson['timeTake']);//新建table
}


  

//根据获取的数据，创建table
  //创建table
function createTable(result,timeTake){
        //根据获取的数据，创建table
        jQuery("#mytable").empty();
        var jsObject = result;
        //var jsObject = [{"name":10,"value":20},{"name":10,"value":20}];
        var p = "<h3>耗时:"+timeTake+"秒 ,识别结果为:</h3>";
        var tableString =p+ "<table id='billmodeltable' class='gridtable'><tr><th>序号</th><th>值</th></tr>"
                        
        for(var i=0;i<jsObject.length;i++){
            tableString+="<tr><td><p>"+i+"</p></td><td><p contenteditable='true'>"+jsObject[i]["text"]+"</p></td></tr>";
        }
        tableString+="</table>";
        //jQuery("#mytable").append(p);
        jQuery("#mytable").append(tableString);
        
    }
        
    
