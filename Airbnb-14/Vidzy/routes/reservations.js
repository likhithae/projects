var express = require('express');
var router = express.Router();
var monk = require('monk')
var db = monk('localhost:27017/vidzy');
var collection = db.get('reservations');

router.get('/', function (req, res) {
    collection.find({}, function (err, reservations) {
        if (err) throw err;
        res.json(reservations);
    });
});


 /*
    {
        "check_in_date":"2022-11-21",
        "check_out_date":"2022-11-24",
        "guest_id":"1",
        "property_id":"6373feccbb331bfb9b5b0dd2",
        "number_of_guests":{
            "adults":2,
            "children":1
        },
        "booking_date":"2022-11-10",
        "booking_status":"completed",
        "price":"$300",
        "long_stay_discount":0,
        "guest_payment_details":"NULL",
        "payment_method":"NULL",
        "payment_date":"NULL",
        "payment_status":"pending"
    }
    */