/*
	Author: Matt Murray
	Client: Accenture
	Project: Eida
	Description: the core module that other modules can inherit
*/

var fs = require('fs'),
  path = require('path'),
  utils = require('util'),
  eidaUtils = require('../eida_utils.js');

var natural = require('natural');
var jd = require('jsdataframe');
var _ = require('underscore');


//TODO:
// [] Get sort to work
// [] Check against py script
var CAHBot = function (_corpus)
{
	console.log("setting up CAH bot");
	var _this = this;
    console.log(Object.keys(_corpus));
    this.CachedVecFile = path.join(__dirname, '../corpora/cah_vectors.json');
    this.CachedVectors = _corpus['cah_vectors'];
    this.Cards = _corpus['cah_cards'];

	// Reads GloveVecFile and stores vectors in GloveVectors
    this.parseGloveVectors = function (_glov){
    	console.log("parsing the oh so large glove vectors");
    	var allText = _glov.split('\n');
    	var gv = {};
	    for (var line = 0; line < allText.length; line++){
	      splitLine = allText[line].split(' ')
	      word = splitLine[0]
	      vector = [];
	      for (var num = 1; num < splitLine.length; num++){
	        vector.push(parseFloat(splitLine[num]))
	      }
	      gv[word] = vector
	    }
	    console.log("Done." + Object.keys(gv).length + " words loaded!");
	    return gv;
    }
    this.GloveVectors = this.parseGloveVectors(_corpus['glove_6B_50d']);
    

    this.addWordsToCache = function(word, vector){
    	console.log("adding: " +word + " to cache")
        _this.CachedVectors[word] = vector;
        /*
        fs.writeFile(_this.CachedVecFile, JSON.stringify(_this.CachedVectors),function(err){
            if (err) {
                throw err
            }
            console.log("cached vec file saved");
        });
        */
    };

    this.createSentenceVector = function(text){
        // Params: sentence string
        // Returns: matrix of word vectors
        console.log("creating sentence vec of: " +text);
        var tokenizer = new natural.WordTokenizer()
        var stopwords = natural.stopwords
        var base_folder = path.join(path.dirname(require.resolve("natural")), "brill_pos_tagger");
        var rulesFilename = base_folder + "/data/English/tr_from_posjs.txt";
        var lexiconFilename = base_folder + "/data/English/lexicon_from_posjs.json";
        var defaultCategory = 'N';

        var lexicon = new natural.Lexicon(lexiconFilename, defaultCategory);
        var rules = new natural.RuleSet(rulesFilename);
        var tagger = new natural.BrillPOSTagger(lexicon, rules);

        var tokens = tokenizer.tokenize(text)
        var vectors = []
        var vecKeys = Object.keys(_this.CachedVectors);
        console.log(vecKeys);
        return jd.rep(0,50);

        for (var i = 0; i < Object.keys(tokens).length; i++ ){
            token = tokens[i].toLowerCase()
            if (stopwords.indexOf(token) == -1){
                if (vecKeys.indexOf(token) != -1){
                    if (tagger.tag([token])[1] == 'N' || tagger.tag([token]) == 'VB'){
                        var vector = _this.CachedVectors[token].map(function(x) {x * 1.5})
                        vectors.push(vector)
                    }
                    else{
                        vectors.push(_this.CachedVectors[token])
                    }
                }
                else {
                    if (_this.GloveVectors.length == 0){
                        _this.readGloveVectors()
                    }
                    if (token in this.GloveVectors){
                        vector = this.GloveVectors[token]
                        vectors.push(vector)
                        _this.addWordsToCache(token, vector)
                    }
                }
            }
        }
        //If vectors is empty (no words found), return vector of zero vec dim (50,0)
        if (vectors.length == 0){
            return jd.rep(0,50)
        }
        return vectors
    };

    this.createAverageVector = function(sentence_mat){
        // Params: matrix of word vectors
        // Returns: average vector of sentence matrix
        sum_vec = jd.rep(0,50)
        for (var i = 0; i < sentence_mat.length; i++ ){
            var vec = sentence_mat[i]
            sum_vec = sum_vec.add(vec)
        }

        avg_vec = sum_vec.div(sentence_mat.length)
        return avg_vec
    };

    this.cosineSimilarity = function(v1, v2){
        // compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        var sumxx = 0
        var sumxy = 0
        var sumyy = 0
        for ( var i = 0; i < 50; i++ ){
            var x = v1[i]
            var y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        }
        return sumxy/Math.sqrt(sumxx*sumyy)
    };

    //Pick Best card
    this.pickBestCard = function(wc_array, black_card){
        // """
        // Params: Array of white card dicts, black card text
        // in the format {text: string, pick: int}
        // Returns: dict of best card texts, average vectors, and confidence scores
        //
        // If wc_array = None, pick best white card from entire set
        // """
        console.log("picking best card");
        var wc_vectors = {}
        // Creates an array of word vectors for each wc text
        for (var i = 0; i < wc_array.length; i++ ){
            var wc_dict = wc_array[i]
            wc_vectors[wc_dict.text] = _this.createSentenceVector(wc_dict.text)
        }

        // Creates list of vectors for black card text
        var bc_vector = _this.createSentenceVector(black_card)

        // Iterates through each cards vectors. Adds each array elementwise and
        // and takes the average.
        var avg_dict = {}

        for (var i = 0; i < wc_array.length; i++ ){
            var wc_dict = wc_array[i]
            avg_dict[wc_dict.text] = _this.createAverageVector(wc_vectors[wc_dict.text])
        }
        // Create black card avg vector
        var bc_avg_vector = this.createAverageVector(bc_vector)
        // Create cosine similarity scores for each prompt/response combination
        var prompt = black_card
        var prompt_vec = bc_avg_vector.values
        var scores = [] //{text:score}
        for (var i = 0; i < Object.keys(avg_dict).length; i++ ){
            var key = Object.keys(avg_dict)[i]
            var response_vec = avg_dict[key].values
            var similarity = this.cosineSimilarity(prompt_vec, response_vec)
            var abs_distance_from_zero = parseFloat(Math.abs(similarity))
            //console.log(key)
            var score = {}
            score[key] = abs_distance_from_zero
            scores.push(score)
        }

        //Sort array of dicts by values ascending
        //TODO: get sort to work
        scores.sort(function(a,b){
            return a.value - b.value
        })
        //console.log(scores)
        return {best: Object.keys(scores[0]), options: scores}
    };

    this.formResponse = function(black_card, wc_array){
        // """
        // Params: array of white card dicts, black_card dict, num picks
        // Returns: Complete sentence replacing '_' in black card text with
        //          responses
        // """

        if (wc_array == undefined){
            wc_array = _this.Cards.whiteCards
        }
        var wc = [];
        var result = black_card;
        while (result.indexOf('_') != -1){
        	var bc = _this.pickBestCard(wc_array, black_card);
            var wc_text = bc.best;
            //console.log(wc_text)
            //wc.push(bc.options);
            wc = bc.options;
            // Get dict to pop later
            var wc_dict = {}
            wc_array.forEach(function(item){
                if (item.text == wc_text){
                    wc_dict = item
                }
            })
            // Clean text
            wc_dict.text = wc_dict['text'].replace(/[^\w\s]|_/g, "")
            if(wc_dict.keepCap == false){
                wc_dict.text = wc_dict['text'].toLowerCase()
            }
            result = result.replace('_', wc_dict.text)
            wc_array.pop(wc_array.indexOf(wc_dict));
        }

        return {blackCard: black_card, whiteCards: wc, result: result}
    };

    this.simulateRound = function(pick){
        
        var cards = _this.Cards
        // Get random 7 response cards
        var wc_array = _.sample(_this.Cards.whiteCards, 7)
        // Get 1 random black card with pick = x
        var pick_x = []
        _this.Cards.blackCards.forEach(function(card){
            if (card.pick == pick){
                pick_x.push(card)
            }
        })
        var black_card = _.sample(pick_x, 1)[0]['text']

        if (black_card.indexOf('_') == -1){
            black_card += ' _'
        }

        var response = _this.formResponse(black_card, wc_array)
        return response
    };

}



var CAHModule = function(_core){

	this.core = _core;
	this.flavor = "CAHModule";
	this.core.logger.info(this.flavor + " is LOADING");
	this.socket = null;
	this.bot = null;
	var _this = this;

	this.apiMap = {
		test:"/api/"+this.flavor+"/test"
	};

	//default io mapping for socket events
	//{eventName: "", eventHandler: (function)}
	this.ioMap = {
		oneLiner:'one liner',
		blackCard:'black card',
		pythonOneLiner:'python one liner',
		pythonBlackCard:'python black card'
	};


	this.testHandler = function(req, res){
		_this.core.logger.log("I have a belly button");
		res.send("OMG somthing worked");
	}

	this.handlePythonOneLiner = function(msg){
		console.log(msg);
		_this.core.voice.vocalize(msg.text, function(adata){
			console.log("got audio data");
			//getting message back from python connection
			_this.socket.emit(_this.ioMap.oneLiner, {text: msg.text, audio: adata});
			
		});
		
	}

	this.handlePythonBlackCard = function(msg){
		console.log(msg);
		_this.core.voice.vocalize(msg.text, function(adata){
			console.log("got audio data");
			//getting message back from python connection
			_this.socket.emit(_this.ioMap.blackCard, {text: msg.text, audio: adata});
			
		});
		
	}

	this.handleOneLiner = function(msg){
		//console.log(_this.core.pythonConnection.id);
		//got message from user to get one liner
		var rndOne = _this.bot.simulateRound(1);
		console.log(rndOne);
		
		_this.core.voice.vocalize(rndOne.result, function(adata){
			console.log("got audio data");
			//getting message back from python connection
			_this.socket.emit(_this.ioMap.oneLiner, {text: rndOne.result, audio: adata});
			
		});

		//tell python socket to send one liner
		//_this.core.pythonConnection.emit(_this.ioMap.pythonOneLiner);
	}

	this.handleBlackCard = function(msg){
		console.log("handling new black card: " + msg);
		//console.log(_this.core.pythonConnection.id)
		//got message from user to get one liner
		var wc = _.sample(_this.bot.Cards.whiteCards, 7);
		var newBC = _this.bot.formResponse(msg, wc); // <-- this takes too long if not limited to small set like 7
		//need to cache sentence vectors!!!
		
		_this.core.voice.vocalize(newBC.result, function(adata){
			console.log("got audio data");
			//getting message back from python connection
			_this.socket.emit(_this.ioMap.oneLiner, {text: newBC.result, audio: adata});
			
		});

		//tell python socket to send one liner
		//_this.core.pythonConnection.emit(_this.ioMap.pythonBlackCard, msg);
	}

	this.core.apiCalls = this.core.apiCalls.concat([
		{
			type:"GET",
			path:this.apiMap.test,
			handler:this.testHandler
		}
	]);

	this.core.socketEvents = this.core.socketEvents.concat([
		{
			eventName:this.ioMap.oneLiner,
			eventHandler:this.handleOneLiner
		},
		{
			eventName:this.ioMap.pythonOneLiner,
			eventHandler:this.handlePythonOneLiner
		},
		{
			eventName:this.ioMap.blackCard,
			eventHandler:this.handleBlackCard
		},
		{
			eventName:this.ioMap.pythonBlackCard,
			eventHandler:this.handlePythonBlackCard
		}
	]);

};


function setup(){
	this.bot = new CAHBot(this.core.brain.corpora.data);
	
}

function attachSocket(_socket){
	this.socket = _socket;
	//console.log("attaching socket", this.socket)
}


CAHModule.prototype.setup = setup;
CAHModule.prototype.attachSocket = attachSocket;



/*
Export
*/
module.exports = CAHModule;




