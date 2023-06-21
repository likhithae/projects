var express = require('express');
var router = express.Router();
tr = require('transliteration').transliterate;
//var flash = require('req-flash');

const bcrypt = require('bcryptjs')
const jwt = require('jsonwebtoken')
const authenticate = require('./middleware/authenticate');
const upload = require('./middleware/upload');
var monk = require('monk');
const { response } = require('express')
var db = monk('localhost:27017/vidzy');
var collection_properties = db.get('properties');
var collection_reservations = db.get('reservations')
var collection_users = db.get('users');
var collection_host = db.get('host');
var collection_users = db.get('users');
var collection_reviews = db.get('reviews');

var user_details = ''
try {user_details = req.cookies["user_details"];}
catch {user_details = ''}

router.get('/', function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	if (!user_details){user_details =''}
	console.log(user_details)
	collection_properties.find({ status: 'available','host_id': {$ne : user_details}}, function (err, properties) {
		if (err) throw err;
		var msg_search = ''
		try {msg_search = req.cookies["msg_search"];}
		catch {msg_search = ''}
		res.clearCookie("msg_search", { httpOnly: true });

		if (user_details){
			collection_users.find({ _id: user_details}, function (err, result) {
				var user_favourites=[]
				result[0].favourites.forEach(function(favourite){
					user_favourites.push(favourite.property_id)
				})
				if (err) throw err;
				res.render('home', { properties: properties, message: msg_search,user_details:user_details,user_favourites:user_favourites });
		});
		}
		else{
			res.render('home', { properties: properties, message: msg_search,user_details:user_details,user_favourites:'' });
		}
		
	});
});

/*
router.get('/properties', function (req, res) {
  collection_properties.find({status:'available'}, function (err, properties) {
	  if (err) throw err;
	  res.render('index',{properties:properties});
  });
}); 
*/

router.post('/properties', upload.fields([{ name: "image", maxCount: 1, }, { name: "all_photos", maxCount: 30, }]), authenticate, function (req, res) {

	let photo = ''
	let all_photo = []
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}

	if (typeof req.body.amenities === 'string') {
		req.body.amenities = [req.body.amenities]
	}

	if (req.files.image) {
		req.files.image.forEach(function (files, index, arr) {
			photo = files.path.replace('public', '')
			all_photo.push(photo)
		})
	}

	if (req.files.all_photos) {
		req.files.all_photos.forEach(function (files, index, arr) {
			all_photo.push(files.path.replace('public', ''))
		})
	}

	if (photo == '') {
		photo = '/images/img1.jpeg'
		all_photo = [photo]
	}
	if (all_photo.length == 0) {
		all_photo = [photo]
	}

	var availability = {
		from: req.body.availability_from,
		to: req.body.availability_to
	};


	collection_properties.insert({
		id: req.body.num,
		title: req.body.title,
		city: req.body.city,
		description: req.body.desc,
		nightly_fee: req.body.price,
		cleaning_fee: req.body.cleaning,
		service_fee: req.body.service,
		amenities: req.body.amenities,
		bedrooms: req.body.bedrooms,
		beds: req.body.beds,
		photo: photo,
		all_photos: all_photo,
		rating: 0,
		num_rating: 0,
		reviews: [],
		favourite: false,
		host: req.body.host,
		host_id: req.user.user_id,
		full_description: req.body.fulldesc,
		capacity: req.body.capacity,
		property_type: req.body.property_type,
		cancellation_policy: req.body.cancellation_policy,
		house_rules: req.body.house_rules,
		location: {
			street: req.body.street,
			city: req.body.city,
			state: req.body.state,
			zipcode: req.body.zipcode,
			country: req.body.country,
		},
		availability: [availability],
		status: 'available'

	}, function (err, result) {
		if (err) throw err;
		// if insert is successfull, it will return newly inserted object
		//res.json(video);
		collection_host.find({ user_id: req.user.user_id }, function (err, result1) {
			var properties = result1[0].properties
			properties.push(result._id)

			collection_host.update(
				{ "user_id": req.user.user_id },
				{ "$set": { "properties": properties } }
			);
		});

		res.redirect('/property_listing');
	});
});

router.get('/properties/new', function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	res.render('new',{user_details:user_details});
});

router.get('/properties/:id', function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_properties.find({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		var user = ''
		var host = false
		if (req.user) {
			user = req.user.user_id;
		}
		if (user_details){
			collection_users.find({ _id: user_details}, function (err, result1) {
				var user_favourites=[]
				result1[0].favourites.forEach(function(favourite){
					user_favourites.push(favourite.property_id)
				})
				if (err) throw err;
				console.log("ok")
				res.render('show', { property: result[0], user: user, host: host,user_details:user_details,user_favourites:user_favourites });
		});
		}
		else{
			res.render('show', { property: result[0], user: user, host: host,user_details:user_details,user_favourites:'' });
		}
		//res.render("test", {apartment: result[0], num_days: 2, guests:2, check_in: '', check_out: '', booked: false });
		//res.json(result);
	});
});

router.get('/:host_id/properties/:id', authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_properties.find({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		var user = ''
		var host = false
		if (req.user) {
			user = req.user.user_id;
			collection_users.find({ _id: req.user.user_id }, function (err, result1) {
				if (err) throw err;
				host = result1[0].is_host
				res.render('show', { property: result[0], user: user, host: host,user_details:user_details,user_favourites:'' });
				
			});
		}

		//res.render("test", {apartment: result[0], num_days: 2, guests:2, check_in: '', check_out: '', booked: false });
		//res.json(result);
	});
});

router.delete('/properties/:id', function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	//collection_properties.remove({ _id: req.params.id }, function(err, video){
	var set_status = 'unavailable';
	if (req.query.value.toString() == '2') {
		console.log("yass")
		set_status = 'available';
	}
	console.log(set_status)
	collection_properties.update({ _id: req.params.id }, {
		$set: {
			status: set_status
		}
	}, function (err, result) {
		if (err) throw err;
		res.redirect('/property_listing');
		//res.json(video);
	});
});

router.get('/properties/:id/edit', function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_properties.find({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		res.render('edit', { property: result[0], message: '0' ,user_details:user_details});
	});
});

router.get('/properties/:id/review', authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_properties.find({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		res.render('review', { property: result[0], message: '0',user_details:user_details });
	});
});

router.post('/properties/:id/review', authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_properties.find({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		var review = {
			"user_id": req.user.user_id,
			"username": req.body.username,
			"rating": req.body.rating,
			"comment": req.body.comment,
			"date_posted": new Date()
		}
		var reviews = result[0].reviews
		var rating = parseFloat(result[0].rating) + parseFloat(req.body.rating)
		var num_rating = parseInt(result[0].num_rating) + 1
		//rating = parseFloat(rating/num_rating)
		reviews.push(review)
		collection_reviews.insert({
			user_id: req.user.user_id,
			property_id: req.params.id,
			username: req.body.username,
			rating: req.body.rating,
			comment: req.body.comment,
			date_posted: new Date()
		});

		collection_properties.update(
			{ "_id": req.params.id },
			{ "$set": { "reviews": reviews, "rating": rating, "num_rating": num_rating } }
		);

		res.redirect('/properties/'+req.params.id);
	});
});

router.put('/properties/:id/edit', upload.fields([{ name: "image", maxCount: 1, }, { name: "all_photos", maxCount: 30, }]), function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_properties.find({ _id: req.params.id }, function (err, result) {
		let property = result[0]
		let photo = ''
		let all_photo = []
		if (typeof req.body.old_photos === 'string') {
			all_photo = req.body.old_photos.split(",");
		}

		if (req.files.image) {
			req.files.image.forEach(function (files, index, arr) {
				photo = files.path.replace('public', '')
				all_photo.push(photo)
			})
		}

		if (req.files.all_photos) {
			req.files.all_photos.forEach(function (files, index, arr) {
				all_photo.push(files.path.replace('public', ''))
			})
		}
		if (photo == '') {
			photo = req.body.old_image
		}
		if (typeof req.body.amenities === 'string') {
			req.body.amenities = [req.body.amenities]
		}

		var today = new Date();
		var dd = String(today.getDate()).padStart(2, '0');
		var mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
		var yyyy = today.getFullYear();
		today = yyyy + '-' + mm + '-' + dd;

		var all_dates = [];
		var entries = [];
		var d1 = '', d2 = '', from = '', to = '';

		// For each from-to date pair
		if (req.body.availability){
			entries = Object.entries(req.body.availability);
			for (var i = 0; i < property.availability.length; i++) {
				var dates = {
					from: property.availability[i].from,
					to: property.availability[i].to
				};
	
				//console.log(dates.from,entries.find(from => from == 'from_'+i))
				var [from, d1] = entries.find(([from, d1]) => from == "from_" + i);
				var [to, d2] = entries.find(([to, d2]) => to == "to_" + i);
	
				// If any of the availability dates has changed
	
				if (dates.from.valueOf() != d1.valueOf() ||
					dates.to.valueOf() != d2.valueOf()) {
	
					if (d1.valueOf() < today.valueOf() || d2.valueOf() < today.valueOf() ||
						d1.valueOf() > d2.valueOf()) {
						//req.flash("error", "Availability dates should be valid. Please try again.");
						return res.render("edit", { property: property, message: '1',user_details:user_details });
					};
	
					if (dates.from.valueOf() != d1.valueOf()) {
						dates.from = d1;
					};
	
					if (dates.to.valueOf() != d2.valueOf()) {
						dates.to = d2;
					};
				};
	
				all_dates.push(dates);
			};
		}
		

		// If new dates have been added
		if (req.body.more_dates_from && req.body.more_dates_to) {
			dates = {
				from: req.body.more_dates_from,
				to: req.body.more_dates_to
			};
			all_dates.push(dates);
		};

		if (all_dates.length > 0) {
			all_dates.sort(function ({ from: f1, to: t1 }, { from: f2, to: t2 }) {
				var d1 = Date.parse(f1),
					d2 = Date.parse(f2),
					one_day = 1000 * 60 * 60 * 24,
					diff = Math.round((d2 - d1) / one_day);
				return diff < 0;
			});
		};

		collection_properties.update({ _id: req.params.id }, {
			$set: {
				id: req.body.num,
				title: req.body.title,
				city: req.body.city,
				description: req.body.desc,
				nightly_fee: req.body.price,
				cleaning_fee: req.body.cleaning,
				service_fee: req.body.service,
				amenities: req.body.amenities,
				bedrooms: req.body.bedrooms,
				beds: req.body.beds,
				photo: photo,
				all_photos: all_photo,
				favourite: false,
				host: req.body.host,
				full_description: req.body.fulldesc,
				capacity: req.body.capacity,
				property_type: req.body.property_type,
				cancellation_policy: req.body.cancellation_policy,
				house_rules: req.body.house_rules,
				location: {
					street: req.body.street,
					city: req.body.city,
					state: req.body.state,
					zipcode: req.body.zipcode,
					country: req.body.country
				},
				availability: all_dates,
				status: 'available'


			}
		}, function (err, result) {
			if (err) throw err;
		});
		collection_properties.find({ _id: req.params.id }, function (err, result) {
			if (err) throw err;
			res.redirect('/'+property.host_id+'/properties/' + req.params.id);
		});
	});

});

router.get('/reservations/list/:id', authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_reservations.find({ guest_id: req.params.id.toString() }, function (err, result) {
		if (err) throw err;
		collection_users.find({ _id: req.params.id.toString() }, function (err, result1) {
			if (err) throw err;
			res.render('show_res', { reservations: result, user: result1[0],user_details:user_details });
		});
	});
});

router.get('/reservations/:id/new', authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_properties.find({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		let dates_valid = ''
		try {
			dates_valid = req.cookies["dates_valid"];
		}
		catch {
			dates_valid = ''
		}
		res.clearCookie("dates_valid", { httpOnly: true });
		res.render('new_res', { guest: req.user.user_id.toString(), properties: result[0], message: dates_valid,user_details:user_details });
	});
});

router.get('/reservations/:id/details', function (req, res) {
	var properties
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_reservations.find({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		collection_properties.find({ _id: result[0].property_id }, function (err, properties) {
			if (err) throw err;
			var delete_res = false
			var date1 = result[0].check_in_date
			const then = new Date(date1);
			const now = new Date();
			const msBetweenDates = (then.getTime() - now.getTime());
			const hoursBetweenDates = msBetweenDates / (60 * 60 * 1000);
			console.log(result[0])
			if (hoursBetweenDates < 0) {
				delete_res = false
			}
			else if (hoursBetweenDates < 48 & hoursBetweenDates > 0) {
				delete_res = false
			}
			else {
				delete_res = true
			}
			res.render('details_res', { reservation: result[0], properties: properties[0], delete_res: delete_res,user_details:user_details });
		});
	});
});

router.delete('/reservations/:id/delete', function (req, res) {
	var guest
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_reservations.find({ _id: req.params.id }, function (err, result) {
		guest = result[0].guest_id
	});
	collection_reservations.remove({ _id: req.params.id }, function (err, result) {
		if (err) throw err;
		res.redirect('/reservations/list/' + guest);
	});
});


router.get('/register', function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	var context = ''
	try {context = req.cookies["context"];}
	catch {context = ''}

	res.clearCookie("context", { httpOnly: true });
	res.render('register', { message: context,user_details:user_details });
});

router.post('/register', upload.single('profile_photo_url'), function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	const { first_name, last_name, date_of_birth, email_id, phone_number, password } = req.body;
	const is_guest = true;
	const is_host = false;
	const date_registered = new Date();
	const favourites = [];

	if (req.file) {
		profile_photo_url = req.file.path.replace('public', '')
	}
	else {
		profile_photo_url = "/uploads/default_profile.jpg"
	}

	if (!(first_name && last_name && email_id && password)) {
		res.cookie("context", "2", { httpOnly: true });
		res.redirect('/register');
	}
	else if (req.body.password != req.body.passwordrepeat) {
		req.flash("error", "Password confirmation failed. Please try again.");
		res.cookie("context", "3", { httpOnly: true });
		res.redirect('/register');
	}
	else {
		collection_users.findOne({ email_id: email_id }, function (err, user) {
			if (err) throw err;

			if (user) {
				//res.json({ error : "User already exists. Please login!"} );
				req.flash('message', 'This email already exists');
				res.cookie("context", "1", { httpOnly: true });
				res.redirect('/register');
			}
			else {
				bcrypt.hash(req.body.password, 10, function (err, hashed_password) {
					if (err) {
						res.json({
							error: err
						})
					}
					let newUser = {
						first_name,
						last_name,
						email_id,
						hashed_password,
						phone_number,
						date_of_birth,
						date_registered,
						profile_photo_url,
						is_guest,
						is_host,
						favourites
					}
					collection_users.insert(newUser, function (err, user) {
						if (err) throw err;
						var token = jwt.sign({ user_id: user._id, email_id }, process.env.ACCESS_TOKEN_SECRET);
						if (token) {
							user.token = token;
						}
						res.redirect('/login');
					})
				});
			}

		});

	}



});

router.get('/login', function (req, res) {
	var context1 = ''
	try {context1 = req.cookies["context1"];}
	catch {context1 = ''}
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	res.clearCookie("context1", { httpOnly: true });
	res.render('login', { message: context1,user_details:user_details });
});

router.post('/login', function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	const { email_id, password } = req.body;

	if (!(email_id && password)) {
		res.cookie("context1", "2", { httpOnly: true });
		res.redirect('/login');
	}
	else {

		collection_users.findOne({ email_id: email_id }, function (err, user) {
			if (err) throw err;
			if (user == null) {
				res.cookie("context1", "1", { httpOnly: true });
				res.redirect('/login');
			}
			else {
				bcrypt.compare(password, user.hashed_password, function (err, result) {
					if (err) {
						res.cookie("context1", "3", { httpOnly: true });
						res.redirect('/login');
					}

					if (result) {
						let token = jwt.sign({ user_id: user._id, email_id }, process.env.ACCESS_TOKEN_SECRET, { expiresIn: process.env.ACCESS_TOKEN_EXPIRE_TIME });
						let refresh_token = jwt.sign({ user_id: user._id, email_id }, process.env.REFRESH_TOKEN_SECRET, { expiresIn: process.env.REFRESH_TOEKEN_EXPIRE_TIME });

						user.token = token;
						user_details = user._id
						res.cookie("user_details", user._id, {
							httpOnly: true,
							maxAge: 3 * 60 * 60 * 1000, // 3hrs in ms
						});
						res.cookie("jwt", token, {
							httpOnly: true,
							maxAge: 3 * 60 * 60 * 1000, // 3hrs in ms
						});
						res.cookie("jwt_refresh", refresh_token, {
							httpOnly: true,
							maxAge: 3 * 60 * 60 * 1000, // 3hrs in ms
						});
						//res.json(user);
						//res.header("x-auth-token", token).send({
						//	_id: user._id,
						//	email: user.email_id
						//  });
						res.redirect('/')
					}
					else {
						res.cookie("context1", "3", { httpOnly: true });
						res.redirect('/login');
					}

				})
				/*
				if (user.password === password ){
					var token = jwt.sign({ user_id: user._id, email_id}, process.env.ACCESS_TOKEN_SECRET);
					user.token = token;
					//res.json(user);
					res.redirect('/')
				}
				else{
					res.cookie("context1", "3", { httpOnly: true });
					res.redirect('/login');
				}
				*/

			}

		});

	}

});

/*
app.get("/",(req,res) => {
	const refreshtoken = req.cookies.jwt_refresh
	jwt.verify(refreshtoken,process.env.REFRESH_TOKEN_SECRET, function(err,decode){
		if(err){
			res.status(400).json({
				err
			})
		}
		else{
			let token = jwt.sign({name:decode.name},process.env.ACCESS_TOKEN_SECRET,{expiresIn:process.env.ACCESS_TOKEN_EXPIRE_TIME})
			let refreshtoken = req.cookies.jwt_refresh
			res.status(200).json({
				message:"Token refreshed succesfully",
				token,
				refreshtoken
			})
		}
	})
}) 
*/

router.get("/logout", (req, res) => {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	res.cookie("jwt", "", { maxAge: "1" })
	res.cookie("user_details", "", {maxAge: "1" })
	res.redirect("/")
	user_details = ''
})

router.post("/search", function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	
	var location = req.body.location,
		check_in = req.body.check_in,
		check_out = req.body.check_out,
		guests = req.body.guests,
		apartments = [];

	try {
		location = tr(location);
	}
	catch (error) {
		res.cookie("msg_search", "Please enter a proper location", { httpOnly: true });
		return res.redirect("/");
	}

	//collection_properties.find({}).populate("host").populate("reservations").populate("reviews")
	collection_properties.find({ status: 'available','host_id': {$ne : user_details} }, function (err, results) {
		if (err) {
			res.cookie("msg_search", err, { httpOnly: true });
			return res.redirect("/");
		} else {
			// Check if the renting dates are valid
			var today = new Date();
			var dd = String(today.getDate()).padStart(2, '0');
			var mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
			var yyyy = today.getFullYear();
			today = yyyy + '-' + mm + '-' + dd;
			if (check_in.valueOf() < today.valueOf() ||
				check_out.valueOf() < today.valueOf() ||
				check_in.valueOf() > check_out.valueOf()) {
				res.cookie("msg_search", "Dates should be valid. Please try again.", { httpOnly: true });
				return res.redirect("/");
			}

			var d1 = Date.parse(check_in),
				d2 = Date.parse(check_out),
				one_day = 1000 * 60 * 60 * 24,
				diff = Math.round((d2 - d1) / one_day);


			results.forEach(function (apartment) {

				location = location.replace(',', '').replace('.', '').toLowerCase();
				city = apartment.location.street + apartment.location.city + apartment.location.state + apartment.location.country + apartment.location.zipcode
				city = city.replace(',', '').replace('.', '').toLowerCase();
				var validLocation = 0

				const location_array = location.split(" ");

				for (i in location_array) {
					if (city.includes(location_array[i])) {
						validLocation += 1
					}
				}

				//console.log("validk", validLocation, location_array, city)
				for (var dates of apartment.availability) {
					if (check_in.valueOf() >= dates.from.valueOf() &&
						check_out.valueOf() <= dates.to.valueOf() &&
						check_in.valueOf() <= check_out.valueOf() && guests <= parseInt(apartment.capacity) && validLocation >= 1) {
						//diff >= apartment.renting_rules.rent_days_min && 
						apartments.push(apartment);
						break;
					}
				};
			});

			if (apartments.length == 0) {
				res.cookie("msg_search", "No results found for your search.", { httpOnly: true });
				return res.redirect("/");
			}

			var price,
				max_price = 0,
				values = [],
				sorted = [];

			apartments.forEach(function (apartment) {
				values.push([apartment._id, apartment.nightly_fee]);
			});

			values.sort(function ([a, b], [c, d]) { return b - d });
			values.forEach(function ([id, price]) {
				for (var apartment of apartments) {
					if (apartment._id == id) {
						sorted.push(apartment);
						if (parseInt(price.replace('$', '')) > max_price) {
							max_price = price;
						}
						break;
					}
				}
			});

			/*
			var str_apartments = JSON.stringify(sorted);
			var str_location = JSON.stringify(locationObj);
			
			res.redirect(url.format({
				pathname: "/search/page/1",
				query: {
					"apartments": str_apartments,
					"num_days": diff,
					"guests": guests,
					"check_in": check_in,
					"check_out": check_out,
					"str_location": str_location,
					"max_price": max_price
				}
			}));
			*/
			if (user_details){
				collection_users.find({ _id: user_details}, function (err, result) {
					var favourites=[]
					result[0].favourites.forEach(function(favourite){
						favourites.push(favourite.property_id)
					})
					if (err) throw err;
					res.render('search', { properties: sorted, message: '',user_details:user_details,user_favourites:favourites });
			});
			}
			else{
				res.render('search', { properties: sorted, message: '',user_details:user_details,user_favourites:'' });
			}
		}
	});
});

router.get("/favourites/:id", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_users.find({ _id: req.user.user_id }, function (err, result) {
		if (err) throw err;
		res.render('favourites',{user:result[0],user_details:user_details});
});
	/*
	collection_users.find({_id: req.params.id}, function(err, result){
		var properties = ''
		var property= result[0]
		if (property.favourite == true){
			property.favourite = false
			msg = "Property "+property.title + " has been removed from your favourites list"
			res.cookie("msg_favourites", msg, { httpOnly: true });
		}
		else{
			property.favourite = true
			msg = "Property "+ property.title + " has been added to your favourites list"
			res.cookie("msg_favourites", msg, { httpOnly: true });
		}
		var msg_favourites = ''
		try{
			msg_favourites = req.cookies["msg_favourites"];
		} 
		catch{
			msg_favourites = ''
		}
		collection_properties.find({_id: req.user.user_id}, function(err, result){
			//properties = ''//result[0].favourites
			res.render('favourites',{message:msg_favourites,properties:properties})
			//res.status(200).json({
			//message:req.cookies["msg_favourites"],
			//property:req.params.id,
			//user:req.user
			});
		res.render('favourites',{properties:properties})
		});
		*/
});

router.get("/addfavourite", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	var already_added = false
	var fav_added = {
		"property_id": req.query.id,
		"property_title": req.query.title,
		"property_city": req.query.city
	}
	collection_users.find({ _id: req.user.user_id }, function (err, result) {
		if (err) throw err;
		var favourites = []
		result[0].favourites.forEach(function(favourite){
			if (favourite.property_id.toString() == req.query.id){
				already_added = true
			}
			if (favourite.property_id.toString() != req.query.id){
				favourites.push(favourite)
			}
		})
		
		if (already_added == false){
			favourites.push(fav_added)
		}
		collection_users.update(
			{ "_id": req.user.user_id },
			{ "$set": { "favourites": favourites }}
		);
		
	});
	if (req.query.show.toString() == '0'){
		res.redirect('/');
	}
	else{
		res.redirect('/properties/'+req.query.id);
	}
	res.redirect('/');
});

router.get("/removefavourite", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_users.find({ _id: req.user.user_id }, function (err, result) {
		if (err) throw err;
		var favourites = []
		result[0].favourites.forEach(function(favourite){
			if (favourite.property_id.toString() != req.query.id){
				favourites.push(favourite)
			}
		})

		collection_users.update(
			{ "_id": req.user.user_id },
			{ "$set": { "favourites": favourites }}
		);
	});
	res.redirect('/favourites/'+req.user.user_id);
});

router.get("/account", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_users.find({ _id: req.user.user_id }, function (err, result) {
		if (err) throw err;
		if (result[0].is_host == true){
			collection_host.find({user_id:req.user.user_id }, function (err, result1) {
				if (err) throw err;
			res.render('account', { user: result[0],user_details:user_details,host:result1[0] });
			});
		}
		else{
			res.render('account', { user: result[0],user_details:user_details,host:'' });
		}
		
	});
})

router.get("/editprofile/:id", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_users.find({ _id: req.user.user_id }, function (err, result) {
		if (err) throw err;
		if (result[0].is_host == true){
			collection_host.find({user_id:req.user.user_id }, function (err, result1) {
				if (err) throw err;
			res.render('edit_profile', { user: result[0],user_details:user_details,host:result1[0] });
			});
		}
		else{
			res.render('edit_profile', { user: result[0],user_details:user_details,host:'' });
		}
	});
})

router.put("/editprofile/:id", authenticate, upload.single('profile_photo_url'), function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	var profile_photo_url = ''
	if (req.file) {
		profile_photo_url = req.file.path.replace('public', '')
	}
	else {
		profile_photo_url = req.body.old_image
	}

	console.log(req.body)
	collection_host.update({"user_id":req.params.id},{"$set":{
		"short_description":req.body.short_description,
		"languages_known":req.body.languages_known,
		"co_hosts":req.body.co_hosts
	}})

	collection_users.update({ _id: req.params.id }, {
		$set: {
			first_name: req.body.first_name,
			last_name: req.body.last_name,
			date_of_birth: req.body.date_of_birth,
			email_id: req.body.email_id,
			phone_number: req.body.phone_number,
			profile_photo_url: profile_photo_url

		}
	}, function (err, result) {
		console.log("ok")
		if (err) throw err;
		collection_users.find({ _id: req.user.user_id }, function (err, result) {
			if (err) throw err;
			if (result[0].is_host == true){
				collection_host.find({user_id:req.user.user_id }, function (err, result1) {
					if (err) throw err;
				res.render('account', { user: result[0],user_details:user_details,host:result1[0] });
				});
			}
			else{
				res.render('account', { user: result[0],user_details:user_details,host:'' });
			}
		});
	});
})

router.post("/reservations/:apartment_id/:tenant_id/add", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	var check_in = new Date(req.body.checkin),
		check_out = new Date(req.body.checkout),
		adults = req.body.adults,
		children = req.body.children

	collection_properties.find({ _id: req.params.apartment_id }, function (err, result) {
		var foundApartment = result[0]
		if (err) {
			req.flash("error", err.message);
			return res.redirect("/");
		}

		// Add new reservation
		// Update apartments availability dates
		var i = 0;
		var available = false
		for (var dates of foundApartment.availability) {
			dates.from = new Date(dates.from)
			dates.to = new Date(dates.to)

			console.log(check_in,dates.from,check_out,dates.to)
			if (check_in.valueOf() >= dates.from.valueOf() && check_out.valueOf() <= dates.to.valueOf()) {
				available = true
				var temp, dd, mm, yyyy, date;
				var newDates = {};

				if (check_in.valueOf() > dates.from.valueOf()) {
					temp = dates.to;

					// Get previous day from check in
					date = new Date(check_in);
					date.setDate(date.getDate());

					// Convert it to a string
					dd = String(date.getDate()).padStart(2, '0');
					mm = String(date.getMonth() + 1).padStart(2, '0'); //January is 0!
					yyyy = date.getFullYear();
					dates.to = yyyy + '-' + mm + '-' + dd;

					if (check_out.valueOf() < temp.valueOf()) {
						// Get next day from check out
						date = new Date(check_out);
						date.setDate(date.getDate() + 1);

						// Convert it to a string
						dd = String(date.getDate()).padStart(2, '0');
						mm = String(date.getMonth() + 1).padStart(2, '0'); //January is 0!
						yyyy = date.getFullYear();

						newDates = {
							from: yyyy + '-' + mm + '-' + dd,
							to: temp
						};

					}
				} else if (check_out.valueOf() < dates.to.valueOf()) {
					// Get next day from check out
					date = new Date(check_out);
					date.setDate(date.getDate() + 1);

					// Convert it to a string
					dd = String(date.getDate()).padStart(2, '0');
					mm = String(date.getMonth() + 1).padStart(2, '0'); //January is 0!
					yyyy = date.getFullYear();

					dates.from = yyyy + '-' + mm + '-' + dd;
				} else {
					var length = foundApartment.availability.length;

					foundApartment.availability = foundApartment.availability.slice(0, i)
						.concat(foundApartment.availability.slice(i + 1, length));
				}

				if (Object.keys(newDates).length > 0) {
					foundApartment.availability.push(newDates);
					foundApartment.availability.sort(function ({ from: f1, to: t1 }, { from: f2, to: t2 }) {
						return f1.valueOf() - f2.valueOf();
					});
				}

				break;
			}
			else {
				available = false
				console.log(check_in.valueOf(), check_out.valueOf(), dates.from.valueOf(),dates.to.valueOf())
			}

			i += 1;
		};

		if (!available){
			res.cookie("dates_valid", "The Selected dates are not valid. Please change the dates", { httpOnly: true });
			return res.redirect('/reservations/' + req.params.apartment_id + '/new')

		}
		
		collection_properties.update(
			{ "_id": foundApartment._id },
			{ "$set": { "availability": foundApartment.availability } }
		);
		//collection_properties.foundApartment.save();

		var num_guests = parseFloat(adults) + parseFloat(children)
		const dateOne = new Date(check_in);
		const dateTwo = new Date(check_out);
		var days = parseFloat(dateTwo.getDate() - dateOne.getDate());
		price = foundApartment.nightly_fee.replace('$', '').replace(' ', '');
		price = parseFloat(price) * num_guests * days;
		collection_reservations.insert({
			check_in_date: new Date(check_in),
			check_out_date: new Date(check_out),
			guest_id: req.params.tenant_id.toString(),
			property_id: req.params.apartment_id.toString(),
			number_of_guests: {
				adults: req.body.adults,
				children: req.body.children
			},
			booking_date: new Date(),
			booking_status: "Completed",
			price: '$' + price,
			long_stay_discount: 0,
			guest_payment_details: "NULL",
			payment_method: "NULL",
			payment_date: "NULL",
			payment_status: "pending"
		}, function (err, reservation) {
			if (err) throw err;
			res.redirect('/reservations/' + reservation._id.toString() + '/details');
		});
	});
})

router.get("/property_listing", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_users.find({ _id: req.user.user_id }, function (err, result) {
		if (err) throw err;
		if (result[0].is_host) {
			collection_properties.find({ host_id: req.user.user_id }, function (err, properties) {
				if (err) throw err;
				res.render('index', { properties: properties, user: req.user.user_id ,user_details:req.user.user_id});
			});
		}
		else {
			res.render('hostsignup', { user: result[0],user_details:user_details });
		}
	});
})

router.post("/hostsignup", authenticate, function (req, res) {
	try {user_details = req.cookies["user_details"];}
	catch {user_details = ''}
	collection_users.update(
		{ "_id": req.user.user_id },
		{ "$set": { "is_host": true } }
	);
	console.log(req)
	collection_host.insert({
		user_id: req.user.user_id,
		short_description: req.body.description,
		languages_known: req.body.languages,
		co_hosts: req.body.cohost,
		is_superhost: false,
		start_date_as_host: new Date(),
		response_rate: '100%',
		response_time: 'within an hour',
		properties: []

	}, function (err, property) {
		if (err) throw err;
		res.redirect('/property_listing')
		//collection_properties.find({}, function (err, properties) {
		//	if (err) throw err;
			//res.render('index', { properties: properties, user: req.user.user_id,user_details:user_details });
		//});
	});
})

module.exports = router;


