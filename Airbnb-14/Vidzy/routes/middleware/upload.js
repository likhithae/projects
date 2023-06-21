const path = require('path')
const multer = require('multer')

var storage = multer.diskStorage({
    destination:function(req, file, cb){
        cb(null, './public/uploads/')
    },
    filename:function(req,file,cb){
        let ext = path.extname(file.originalname)
        cb(null, file.fieldname+Date.now()+ext)
    }
})

var upload = multer({
    storage:storage,
    fileFilter: function(req, file, callback){
        if(file.mimetyoe == "image/png" || file.mimetype == "image/jpg" || file.mimetype == "image/jpeg"){
            callback(null,true)
        }else{
            //console.log('only jpg or png files',file.mimetype)
            callback(null, false)
        }
    },
    limits:{
        fileSize: 1024*1024*2
    }
})

module.exports = upload