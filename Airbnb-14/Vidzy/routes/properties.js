var express = require('express');
var router = express.Router();
var monk = require('monk')
var db = monk('localhost:27017/vidzy');
var collection = db.get('properties');

router.get('/', function (req, res) {
    collection.find({}, function (err, properties) {
        if (err) throw err;
        res.json(properties);
    });
});

router.get('/:id', function (req, res) {
    collection.findOne({_id:req.params.id}, function (err, properties) {
        if (err) throw err;
        res.json(properties);
    });
});

module.exports = router;
