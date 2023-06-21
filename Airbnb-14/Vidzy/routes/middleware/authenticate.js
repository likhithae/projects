const jwt = require('jsonwebtoken');


/*
const verifyToken = (req, res, next) => {

  const token = req.body.token || req.query.token || req.headers["x-access-token"] || req.header("x-auth-token");

//   const authHeader = req.headers["Authorization"];
//   const token = authHeader && authHeader.split(" ")[1]; // Bearer Token

  console.log("token:" + token);

if (!token){

    return res.json({ error: "Token is required for authentication" } );
}
else{

    try{
        var decoded = jwt.verify(token, process.env.ACCESS_TOKEN_SECRET);
        console.log(decoded);
        return next();

    } catch(err){
        
        return res.json({ error: "Invalid token" });
    }
  
    
}

}

module.exports = verifyToken;

*/

const authenticate = (req, res, next) => {
  /*
  try{
      const token = req.headers.authorization.split(' ')[1]
      const decode = jwt.verify(token,process.env.ACCESS_TOKEN_SECRET)
      console.log("--",token,decode)
      req.user = decode
      return next()
  }
  catch(error){
    res.json({
      message: 'Authentication failed!',
      error: error
    })
  }*/

  var token =  req.cookies.jwt//req.headers["authorization"] || req.body.token || req.query.token || req.headers["x-access-token"] || req.header("x-auth-token");
  var user_details = ''

  try{user_details = req.cookies.user_details}
  catch{user_details=''}

if (!token){
    return res.render('login',{message:'4',user_details:user_details} );
}
else{
    try{
        var decoded = jwt.verify(token, process.env.ACCESS_TOKEN_SECRET);
        req.user = decoded
        //console.log("decoded",decoded);
        return next();

    } catch(err){
      if(err.name == "TokenExpiredError"){
        //return res.json({ error: "Token expired" });
        return res.render('login',{message:'4',user_details:user_details} );
      }
        //return res.json({ error: "Invalid token" });
        return res.render('login',{message:'4',user_details:user_details} );
    }
  
    
}


}

module.exports = authenticate