import React, { Component } from 'react';
import {Bar} from 'react-chartjs-2';
import axios from 'axios';





export default  class BarChart extends Component {

  constructor() {
    super()
    this.state = 
      {
        allInfo:'',
        Sex:'',
        GG1:'',
        traveltime:'',
        studytime:'',
        freetime:'',
        absences:'',
        traveltime2:'',
        studytime2:'',
        freetime2:'',
        absences2:'',
        Barchartdata:'',
        BarchartdataSecond:'',
        G2:''
      }
    }

    componentDidMount() {

      axios.get(`http://20.82.112.97:5000/alldata`)
      .then(res => {
        const data = res.data;
        this.setState({allInfo:data})
        
         console.log(this.state.allInfo["0"].school)

        // Create a new array based on current state:
        
        let travelt = [];
        let studyt = [];
        let freet = [];
        let absencesAmount = [];
        let travelt2 = [];
        let studyt2 = [];
        let freet2 = [];
        let absencesAmount2 = [];

        for (var i = 0; i < this.state.allInfo.length; i++) {
              // console.log(this.state.allInfo[i].GG1)
              // retrieving sex
              //sex.push(this.state.allInfo[i].sex );
              // retriving first Grade
              //FirstGrade.push(this.state.allInfo[i].G1 ); 
              if(this.state.allInfo[i].GC1=="pass"){            
              travelt.push(this.state.allInfo[i].traveltime );
              studyt.push(this.state.allInfo[i].studytime );
              freet.push(this.state.allInfo[i].freetime );
              absencesAmount.push(this.state.allInfo[i].absences );
              //SecondGrade.push(this.state.allInfo[i].G2)
              }
              if(this.state.allInfo[i].GC2=="pass"){            
                travelt2.push(this.state.allInfo[i].traveltime );
                studyt2.push(this.state.allInfo[i].studytime );
                freet2.push(this.state.allInfo[i].freetime );
                absencesAmount2.push(this.state.allInfo[i].absences );
                //SecondGrade.push(this.state.allInfo[i].G2)
                }
            }

            // make average
            const average = arr => arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;
    
            travelt = average( travelt );
            studyt = average( studyt );
            freet = average( freet );
            absencesAmount = average( absencesAmount );

            travelt2 = average( travelt2 );
            studyt2 = average( studyt2 );
            freet2 = average( freet2 );
            absencesAmount2 = average( absencesAmount2 );
            const BarchartdataArray=[];
            const BarchartdataArraySecond=[];
            BarchartdataArraySecond.push( travelt2 , studyt2 ,freet2,absencesAmount2  );

            BarchartdataArray.push( travelt , studyt ,freet,absencesAmount  );
            // Set state
          this.setState({ 
             traveltime:travelt, 
             studytime: studyt,
             freetime:freet,
             absences:absencesAmount,
             Barchartdata:BarchartdataArray,
             traveltime2:travelt2, 
             studytime2: studyt2,
             freetime2:freet2,
             absences2:absencesAmount2,
             BarchartdataSecond:BarchartdataArraySecond,
            });

          // console.log(this.state.Barchartdata)


      })
    //   .catch(()=>{
    //     alert('Error retrieving data!!');
    // })
    }



    render() 
    {
      return (

       
        <div> 

         <Bar
        data={{
          labels: ['Traveltime', 'Studytime', 'Freetime', 'Absences'],
          datasets: [
            {
              label: 'Second Exam',
              data: this.state.BarchartdataSecond,
              backgroundColor: [
               
                 'rgba(190,91,104)',
                 'rgba(190,91,104)',
                 'rgba(190,91,104)',
                 'rgba(190,91,104)',
                 'rgba(190,91,104)',

                // 'rgba(255, 206, 86, 0.2)',
                // 'rgba(75, 192, 192, 0.2)',
                // 'rgba(153, 102, 255, 0.2)',
                // 'rgba(255, 159, 64, 0.2)',
              ],
              borderColor: [
                // 'rgba(255, 99, 132, 1)',
                // 'rgba(54, 162, 235, 1)',
                // 'rgba(255, 206, 86, 1)',
                // 'rgba(75, 192, 192, 1)',
                // 'rgba(153, 102, 255, 1)',
                // 'rgba(255, 159, 64, 1)',
              ],
              borderWidth: 1,
            },
//             {
//               label: 'Second Grade',
//               data:  this.state.BarchartdataSecond,
//               backgroundColor:   '#e2975d',
              
//               //borderColor: 'red',
//             },
          ],
        }}
        height={200}
        width={300}
        options={{
          title:{
            display:true,
            text:'Attributes Distribution for students who passed Second Exam'
          },
        //   maintainAspectRatio: false,
          scales: {
            yAxes: [
              {
                ticks: {
                  beginAtZero: true,
                },
              },
            ],
          },
          legend: {
            labels: {
              fontSize: 15,
            },
          },
        }}
      />
    </div>




        
      )
  }
}

