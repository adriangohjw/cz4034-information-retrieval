const { Client } = require('@elastic/elasticsearch')
const client = new Client({ node: 'http://localhost:9200', requestTimeout: 60000})

/*
client.search({
  index: 'parler_users',
  body: {
    aggs: {
      filtered_followers_percentile: {
        filter: {
          range: {
            comments: {
              gte: 1
            }
          }
        },
        aggs: {
          followers_percentile: {
            percentiles: {
              field: "user_followers",
               percents: [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 ]
            }
          }
        }
      }
    }
  }
}, (err, result) => {
  if (err){ 
    console.log(err)
    return;
  }
  console.log(result)
})

client.search({
  index: 'parler_posts',
  body: {
        aggs: {
          impressions_percentile: {
            percentiles: {
              field: "impressions",
               percents: [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 ]
            }
          }
        }
  }
}, (err, result) => {
  if (err){ 
    console.log(err)
    return;
  }
  console.log(result)
})

client.search({
  index: 'parler_posts',
  body: {
        aggs: {
          impressions_percentile: {
            percentiles: {
              field: "upvotes",
               percents: [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 ],
               "hdr": {                                  
                "number_of_significant_value_digits": 5
              }
            }
          }
        }
  }
}, (err, result) => {
  if (err){ 
    console.log(err)
    return;
  }
  console.log(result)
})
*/

var testvalues = {
  '1.0': 0,
  '2.0': 0,
  '3.0': 0,
  '4.0': 0,
  '5.0': 0,
  '6.0': 0,
  '7.0': 0,
  '8.0': 0,
  '9.0': 0,
  '10.0': 0,
  '11.0': 0,
  '12.0': 0,
  '13.0': 0,
  '14.0': 0,
  '15.0': 0,
  '16.0': 0,
  '17.0': 0,
  '18.0': 0,
  '19.0': 0,
  '20.0': 0,
  '21.0': 0,
  '22.0': 0,
  '23.0': 1,
  '24.0': 1,
  '25.0': 1,
  '26.0': 1,
  '27.0': 1,
  '28.0': 1,
  '29.0': 2.0000076293945312,
  '30.0': 2.0000076293945312,
  '31.0': 2.0000076293945312,
  '32.0': 3.0000076293945312,
  '33.0': 3.0000076293945312,
  '34.0': 3.0000076293945312,
  '35.0': 4.000022888183594,
  '36.0': 4.000022888183594,
  '37.0': 5.000022888183594,
  '38.0': 6.000022888183594,
  '39.0': 6.000022888183594,
  '40.0': 7.000022888183594,
  '41.0': 8.000053405761719,
  '42.0': 9.000053405761719,
  '43.0': 10.000053405761719,
  '44.0': 11.000053405761719,
  '45.0': 12.000053405761719,
  '46.0': 14.000053405761719,
  '47.0': 15.000053405761719,
  '48.0': 17.00011444091797,
  '49.0': 19.00011444091797,
  '50.0': 21.00011444091797,
  '51.0': 23.00011444091797,
  '52.0': 26.00011444091797,
  '53.0': 29.00011444091797,
  '54.0': 33.00023651123047,
  '55.0': 37.00023651123047,
  '56.0': 42.00023651123047,
  '57.0': 47.00023651123047,
  '58.0': 54.00023651123047,
  '59.0': 61.00023651123047,
  '60.0': 69.00048065185547,
  '61.0': 79.00048065185547,
  '62.0': 89.00048065185547,
  '63.0': 102.00048065185547,
  '64.0': 117.00048065185547,
  '65.0': 133.00096893310547,
  '66.0': 151.00096893310547,
  '67.0': 172.00096893310547,
  '68.0': 196.00096893310547,
  '69.0': 223.00096893310547,
  '70.0': 254.00096893310547,
  '71.0': 287.00194549560547,
  '72.0': 325.00194549560547,
  '73.0': 368.00194549560547,
  '74.0': 414.00194549560547,
  '75.0': 465.00194549560547,
  '76.0': 522.0038986206055,
  '77.0': 584.0038986206055,
  '78.0': 652.0038986206055,
  '79.0': 725.0038986206055,
  '80.0': 806.0038986206055,
  '81.0': 894.0038986206055,
  '82.0': 996.0038986206055,
  '83.0': 1100.0078048706055,
  '84.0': 1200.0078048706055,
  '85.0': 1400.0078048706055,
  '86.0': 1500.0078048706055,
  '87.0': 1700.0078048706055,
  '88.0': 1900.0078048706055,
  '89.0': 2200.0156173706055,
  '90.0': 2500.0156173706055,
  '91.0': 2800.0156173706055,
  '92.0': 3200.0156173706055,
  '93.0': 3800.0156173706055,
  '94.0': 4400.0312423706055,
  '95.0': 5300.0312423706055,
  '96.0': 6500.0312423706055,
  '97.0': 8300.062492370605,
  '98.0': 11000.062492370605,
  '99.0': 17000.124992370605,
  '100.0': 1200007.9999923706
};



var percentiletocheck = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 ]

//for(var i in Range(1.0.toExponential.apply.))
//console.log(percentileList);

function getSortedReduced(dataList,percentileList){
  var updated = {};
  for(var i = 0; i < percentileList.length;i++){
    var total = percentileList[i];
    var count = 1;
    var currentValue = Math.round(dataList[percentileList[i] + '.0']);
  
    while(currentValue == Math.round(dataList[percentileList[i+1] + '.0'])){
      total += percentileList[i+1];
      count += 1;
      i++;
    }
    updated[total/count] = currentValue;
  }
  const toNumbers = arr => arr.map(Number);
  var keyArray = toNumbers(Object.keys(updated));
  keyArray.sort(function(a, b) {
    return a - b;
  });
  return [keyArray,updated];
}





//console.log(toNumbers(Object.keys(updated)).sort())
/*
for(var key in ordered){
  //same as previous
  // if(previousValue == test[key + '.0']){
  //   stack = stack + key;
  // }else{
  //   if(stack != 0){
  //     //tally
  //   }else{

  //   }

  // }
  // previousValue = test[key + '.0'];
  console.log(key)
  //console.log(updated[key])
}*/


client.search({
  index: 'parler_posts',
  body: {
        aggs: {
          impressions_percentile: {
            percentiles: {
              field: "reposts",
               percents: [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 ],
               "hdr": {                                  
                "number_of_significant_value_digits": 5
              }
            }
          }
        }
  }
}, (err, result) => {
  if (err){ 
    console.log(err)
    return;
  }
  console.log(result)
  var percentile_values = result.body.aggregations.impressions_percentile.values;
  var values = getSortedReduced(percentile_values,percentiletocheck);
  var keyArray = values[0];
  var updated = values[1];
  //keyArray.length
  for(var i = 3; i < 4;i++){
    console.log(updated[keyArray[i]] + " to " + updated[keyArray[i+1]]);

    var queryRange = {range: {
      "reposts": {
        gte: updated[keyArray[i]]
      }
    }}

    if(updated[keyArray[i+1]] != undefined){
      queryRange.range["reposts"].lt = updated[keyArray[i+1]]
    }

    client.updateByQuery({
      index: 'parler_posts',
      conflicts: 'proceed',
      refresh: true,
      
      body: {
        script: {
          lang: 'painless',
          source: 'ctx._source.reposts_score = ' + keyArray[i]
        },
        query: queryRange
      }
    }, (err, result) => {
      console.log(err);
      console.log(result);
    })

  }
})
