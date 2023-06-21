var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var methodOverride = require('method-override');
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
var propertiesRouter = require('./routes/properties');
const oneDay = 1000 * 60 * 60 * 24;
var session = require('express-session');
var flash = require('connect-flash');
var dotenv = require('dotenv')
const multer  = require('multer')
const upload = multer({ dest: './public/uploads/' })

dotenv.config()

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');


app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
app.use(methodOverride('_method'));
//app.use(express.session({ cookie: { maxAge: 60000 }}));
app.use(session({
  secret: "websession",
  saveUninitialized:false,
  cookie: { maxAge: 60000 },
  resave: false 
}));
app.use(flash());
app.use('./public/uploads/',express.static('./public/uploads/'))


//app.use(function(req, res, next){
//  res.locals.currentUser = req.user;
	// If there's anything in the flash, we'll have access to it in any template under var message
//	res.locals.error = req.flash("error");
//	res.locals.success = req.flash("success");
//	res.locals.warning = req.flash("warning");
//	next();
//});

app.use((req, res, next) => {
  res.append('Access-Control-Allow-Origin', ['*']);
  res.append('Access-Control-Allow-Methods', 'GET,PUT,POST');
  res.append('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-auth-token');
  next();
});

app.use('/', indexRouter);
app.use('/users', usersRouter);
app.use('/api/properties', propertiesRouter);


// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
