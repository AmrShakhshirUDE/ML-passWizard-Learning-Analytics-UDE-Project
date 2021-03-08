import React, { Component } from 'react';
import {Bar} from 'react-chartjs-2';
import axios from 'axios';
import { UrlContext } from '../contexts/urlContext';

export default  class BarChartSecond extends Component {

  constructor() {
    super()
    this.state = 
      {
        allInfo:'',
        GG1:'',
        BarchartData:'',
        pass:'',
        fail:'',
        BarchartDataSecond:'',
        BarchartDataThird:'',
        'serverUrl': UrlContext._currentValue,

      }
    }

    componentDidMount() {

      axios.get(this.state.serverUrl + `alldata`)
      .then(res => {
        const data = res.data;
        this.setState({allInfo:data})
        
        // console.log(this.state.allInfo["0"].school)

        // Create a new array based on current state:
        let FirstGradeFemale = [];
        let FirstGradeMale = [];
        let SecondGradeFemale=[];
        let SecondGradeMale = [];
        let ThirdGradeFemale=[];
        let ThirdGradeMale=[];
        
        let BarchartData = [];
        let BarchartDataSecond=[];
        



        for (var i = 0; i < this.state.allInfo.length; i++) {
              // console.log(this.state.allInfo[i].GG1)
              // retrieving sex
              if (this.state.allInfo[i].sex == "F") {
                //if(this.state.allInfo[i].GC1 == "pass"){
                FirstGradeFemale.push(this.state.allInfo[i].G1 );
                SecondGradeFemale.push(this.state.allInfo[i].G2)
                ThirdGradeFemale.push(this.state.allInfo[i].G3)

                //pass = this.state.allInfo[i].pass.length

                //pass.push(pass)
               // }
                
              }
              if (this.state.allInfo[i].sex == "M") {
                //if(this.state.allInfo[i].GC1 == "fail"){
                 // fail = this.state.allInfo[i].fail.length

                 // fail.push(fail)
                  
                //}
                FirstGradeMale.push(this.state.allInfo[i].G1 );
                SecondGradeMale.push(this.state.allInfo[i].G2 );
                ThirdGradeMale.push(this.state.allInfo[i].G3 );


                
              }   
              //GC1.push(this.state.allInfo[i].GC1 );
 
  
            }
            // make average
                const average = arr => arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;
                FirstGradeFemale = average( FirstGradeFemale );
                FirstGradeMale = average( FirstGradeMale );
                SecondGradeFemale = average(SecondGradeFemale);
                SecondGradeMale = average( SecondGradeMale );
                ThirdGradeFemale = average( ThirdGradeFemale );
                ThirdGradeMale = average( ThirdGradeMale );



                //count how many time pass

                BarchartData.push(FirstGradeFemale,SecondGradeFemale,ThirdGradeFemale);
                BarchartDataSecond.push(FirstGradeMale,SecondGradeMale,ThirdGradeMale);
               // BarchartData.push(pass,fail);


            // Set state
          this.setState({ BarchartData:BarchartData, BarchartDataSecond:BarchartDataSecond});

          //console.log(this.state.BarchartData)
          console.log(this.state.allInfo["0"].school)


      }).catch(()=>{
        alert('Error retrieving data!!');
    })
    }



    render() 
    {
      return (

       
        <div> 

         <Bar
        data={{
          labels: ['G1','G2','G3'],
          datasets: [
            {
              label: 'Female',
              data: this.state.BarchartData,
              backgroundColor: [
             
             
              '#ba7d9d',
              '#ba7d9d',
              '#ba7d9d',




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
            {
              label: 'Male',
              data: this.state.BarchartDataSecond,
              backgroundColor:  '#489093',
              borderColor: 'red',
            },
          ],
        }}
        height={200}
        width={300}
        options={{
          title:{
          
              display: true,
              text: 'Three Exam Average Grade Distribution Regarding Gender'
          
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

