// var express = require('express'),
//   app = express(),
//   port = process.env.PORT || 3001;

// app.listen(port);

// console.log('todo list RESTful API server started on: ' + port);

const { Client } = require('@elastic/elasticsearch')
const client = new Client({ node: 'http://localhost:9200'}) //, requestTimeout: 60000s

const querystring = require('querystring');
const express = require('express')
const app = express()
const port = 3000

// app.use(bodyParser.urlencoded({ extended: false }));
// app.use(bodyParser.json());

// /search?search_term=randomstuff&hashtags[]=hashtag1&hashtags[]=hashtag2

// const field_value_skew = 

const functions_list = [
  {"script_score": {
    "script": {
      "params" : {
        "division" : 100,
        "upvote_skew": 3,
        "impressions_skew": 2,
        "followers_skew": 1
      },//"source": "1"
	
      "source": "( " +
        " params.upvote_skew * Math.log(1 + doc['upvotes'].value) + " +
        " params.impressions_skew * Math.log(1 + doc['impressions'].value) + " + 
        " params.followers_skew * Math.log(1 + doc['followers'].value)" +
        ") / params.division"
    }
  }}
]


function elasticSearch(search_term, hashtag = [], query_from = 0, query_size = 20){

    var querySearch = {
        // bool: {
        //     must: SearchHelper.concat_hash_into_array(
        //       SearchHelper.querystring_to_hash(search_term)
        //     ) + SearchHelper::ArrayParam.new('hashtags', hashtags).value,
        //     filter: [],
        //     should: [],
        //     must_not: []
        // },
        "fuzzy": {
            "body": {
                "value": search_term,
                "fuzziness": "AUTO",
                "max_expansions": 50,
                "prefix_length": 0,
                "transpositions": true,
                "rewrite": "constant_score",
                // "weight": 100
            }
        }
    };
    
    if(hashtag.length > 0){
      querySearch = {
        bool:{
          should: querySearch,
          must: {
            "terms": {
              "hashtags": hashtag,
              // "boost": 1.0
            }
          }
        }
      }
    }

     return client.search({
        index: 'parler_posts',
        body: {
            from: query_from,
            size: query_size,
            query: {
              function_score:{
                query: querySearch,
                boost: 10,
                functions: functions_list,
                boost_mode: "sum",
                score_mode: "sum"
              }
            }
        }
      }
      // , (err, result) => {
      //   if (err){ 
      //     console.log(err)
      //     return [];
      //   }else{
      //     console.log(result)
      //     return result.body.hits.hits;
      //   }
      // }
      )
}



app.get('/search', (req, res) => {
    let from = req.query.from;  
    let size = req.query.size;  

    let search_term = req.query.search_term;

    if(typeof search_term != 'string'){
      res.end("Invalid");
      return;
    }
	//console.log(search_term);

    let hashtags = req.query.hashtags;
    if(typeof hashtags == 'string'){
      hashtags = hashtags.replace(/[`~!@#$%^&*()_|+\-=?;:'".<>\{\}\[\]\\\/]/gi,"")
      hashtags = hashtags.split(",");
    }
    // console.log(
      
      elasticSearch(search_term, hashtags, from, size)
      .then((result, err) => {
          if (err){ 
            console.log(err)
            res.end(JSON.stringify({posts:[]}));
          }else{
            res.end(JSON.stringify({posts:result.body.hits.hits}));
          }
	  console.log(result.body.took)
          // console.log(err)
          // console.log(result)
        }).catch((err)=>
          {
            console.log(err);

          })
})



app.get('/suggest', (req, res) => {
  let from = req.query.from;  
  let size = req.query.size;  
  let search_term = req.query.search_term;

  if(typeof search_term != 'string'){
    res.end("Invalid");
    return;
  }

  let hashtags = req.query.hashtags;
  if(typeof hashtags == 'string'){
    hashtags = hashtags.replace(/[`~!@#$%^&*()_|+\-=?;:'".<>\{\}\[\]\\\/]/gi,"")
    hashtags = hashtags.split(",");
  }
  // console.log(
    
    elasticSuggest(search_term, hashtags, from, size)
    .then((result, err) => {
        if (err){ 
          console.log(err)
          res.end(JSON.stringify({posts:[]}));
        }else{
          if(result.body.suggest['suggest-term'].length < 1){
            res.end(JSON.stringify({posts:[]}));
          }
          res.end(JSON.stringify({posts:result.body.suggest['suggest-term'][0].options}));
        }
        // console.log(err)
        // console.log(result)
      }).catch((err)=>
        {
          console.log(err);

        })
})

function elasticSuggest(search_term, hashtag = [], query_from = 0, query_size = 20){

  return client.search({
     index: 'parler_posts',
     body: {
         // from: query_from,
         // size: query_size,
         "suggest": {
           "text" : search_term,
           "suggest-term" : {
             "term" : {
               "field" : "body",
               "size": 3,
               "sort": "frequency",
               "suggest_mode": "always"
             }
           },
           // "my-suggest-2" : {
           //   "text" : "kmichy",
           //   "term" : {
           //     "field" : "user.id"
           //   }
           // }
         }
     }
   }
  //  , (err, result) => {
  //    if (err){ 
  //      console.log(err)
  //      return [];
  //    }else{
  //      console.log(result.body.suggest['suggest-term'])
  //      return result.body.suggest['suggest-term'][0].options;
  //    }
  //  }
   )
 
 }



 app.get('/distribution', (req, res) => {
  // let from = req.query.from;  
  // let size = req.query.size;  

  let search_term = req.query.search_term;
  let fieldname = req.query.field;
  let from = req.query.from;  
  let size = req.query.size;  


  if(typeof search_term != 'string'){
    res.end("Invalid");
    return;
  }

  let hashtags = req.query.hashtags;
  if(typeof hashtags == 'string'){
    hashtags = hashtags.replace(/[`~!@#$%^&*()_|+\-=?;:'".<>\{\}\[\]\\\/]/gi,"")
    hashtags = hashtags.split(",");
  }
  // console.log(
    
    elasticDistribution(search_term, hashtags, fieldname, from, size)
    .then((result, err) => {
        if (err){ 
          console.log(err)
          res.end(JSON.stringify({distribution:[]}));
        }else{
          res.end(JSON.stringify({distribution:result.body.aggregations.result_percentile.values}));
          
        }
        // console.log(err)
        // console.log(result)
      }).catch((err)=>
        {
          console.log(err);

        })
})

function elasticDistribution(search_term, hashtag = [], fieldName = "upvotes", query_from = 0, query_size = 20){

  var querySearch = {
      "fuzzy": {
          "body": {
              "value": search_term,
              "fuzziness": "AUTO",
              "max_expansions": 50,
              "prefix_length": 0,
              "transpositions": true,
              "rewrite": "constant_score"
          }
      }
  };
  
  if(hashtag.length > 0){
    querySearch = {
      bool:{
        should: querySearch,
        must: {
          "terms": {
            "hashtags": hashtag,
            "boost": 1
          }
        }
      }
    }
  }

//   body: {
//     from: query_from,
//     size: query_size,
//     query: {
//       function_score:{
//         query: querySearch,
//         boost: 10,
//         functions: functions_list,
//         boost_mode: "sum",
//         score_mode: "sum"
//       }
//     }
// }

   return client.search({
      index: 'parler_posts',
      body: {
        from: query_from,
        size: query_size,
        query: {
                function_score:{
                  query: querySearch,
                  boost: 10,
                  functions: functions_list,
                  boost_mode: "sum",
                  score_mode: "sum"
                }
              },
        // query:  querySearch,
          aggs: {
            result_percentile: {
              percentiles: {
                field: fieldName,
                 percents: [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 ]
                //  ,"hdr": {                                  
                  // "number_of_significant_value_digits": 5
                // }
              }
            }
          }
          
      }
    }
    // , (err, result) => {
    //   if (err){ 
    //     console.log(err)
    //     return [];
    //   }else{
    //     console.log(result)
    //     return result.body.hits.hits;
    //   }
    // }
    )
}

app.listen(port, '0.0.0.0', () => {
  console.log(`Example app listening at http://localhost:${port}`)
})
