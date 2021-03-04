import {Pie} from 'react-chartjs-2'
import React, { Component } from 'react';
import axios from 'axios';
import { UrlContext } from '../contexts/urlContext';


export default  class PieChart extends Component {
  constructor() {
    super()
    this.state = 
      {
        allInfo:'',
        GG1:'',
        absences:'',
        examFailure:'',
        montherEdu:'',
        fatherEdu:'',
        absences2:'',
        examFailure2:'',
        montherEdu2:'',
        fatherEdu2:'',
        absences3:'',
        examFailure3:'',
        montherEdu3:'',
        fatherEdu3:'',
        Barchartdata:'',
        BarchartdataSecond:'',
        BarchartdataThird:'',
        'serverUrl': UrlContext._currentValue,
      }
    }

    componentDidMount() {

      axios.get(this.state.serverUrl +'alldata')
      .then(res => {
        const data = res.data;
        this.setState({allInfo:data})
        
         console.log(this.state.allInfo["0"].school)

        // Create a new array based on current state:
        let absences= [];
        let examFailure = [];
        let montherEdu = [];
        let fatherEdu =[];
        let absences2= [];
        let examFailure2 = [];
        let montherEdu2 = [];
        let fatherEdu2 =[];
        let absences3= [];
        let examFailure3 = [];
        let montherEdu3 = [];
        let fatherEdu3 =[];

        for (var i = 0; i < this.state.allInfo.length; i++) {
              // console.log(this.state.allInfo[i].GG1)
              // retrieving sex
             
              // retriving first Grade
            
              if(this.state.allInfo[i].GC1 == "pass"){
                montherEdu.push(this.state.allInfo[i].Medu );
                fatherEdu.push(this.state.allInfo[i].Fedu);
              absences.push(this.state.allInfo[i].absences );
              
              if(this.state.allInfo[i].failures=="1"){
              examFailure.push(this.state.allInfo[i].failures );             
              }
             // gradeFirst.push(this.state.allInfo[i].G1);

            }
            if(this.state.allInfo[i].GC2 == "pass"){
              absences2.push(this.state.allInfo[i].absences );
              // retriving first Grade
              examFailure2.push(this.state.allInfo[i].failures );             
              montherEdu2.push(this.state.allInfo[i].Medu );
              fatherEdu2.push(this.state.allInfo[i].Fedu);
             // gradeFirst.push(this.state.allInfo[i].G1);

            }
            // if(this.state.allInfo[i].GC3 == "pass"){
            //   absences3.push(this.state.allInfo[i].absences3 );
            //   // retriving first Grade
            //   examFailure3.push(this.state.allInfo[i].failures3 );             
            //   montherEdu3.push(this.state.allInfo[i].Medu3 );
            //   fatherEdu3.push(this.state.allInfo[i].Fedu3);
            //  // gradeFirst.push(this.state.allInfo[i].G1);

            // }

          }
            examFailure = examFailure.length
            // make average
            const average = arr => arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;
    
            absences = average(  absences );
            // examFailure = average( examFailure);
            montherEdu = average( montherEdu);
            fatherEdu = average( fatherEdu );
            absences2 = average(  absences2 );
            examFailure2 = average( examFailure2);
            montherEdu2 = average( montherEdu2);
            fatherEdu2 = average( fatherEdu2 );
            const BarchartdataArray=[];
            const BarchartdataArraySecond=[];

            BarchartdataArray.push( absences,examFailure,montherEdu,fatherEdu );
            BarchartdataArraySecond.push( absences2,examFailure2,montherEdu2,fatherEdu2)
            // Set state
          this.setState({      
          absences:absences,
          examFailure:examFailure,
          montherEdu:montherEdu,
          fatherEdu:fatherEdu,
          Barchartdata:BarchartdataArray,
          absences2:absences2,
          examFailure2:examFailure2,
          montherEdu2:montherEdu2,
          fatherEdu2:fatherEdu2,
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
    return (<div> 
         <Pie
        data={{
          labels: ['School Absences', 'Exam Failure', 'Mother Education', 'Father Education'],
          datasets: [
            {
              label: 'G1',
              data: this.state.Barchartdata,
              backgroundColor: [
                '#be5168',
                '#3e8e9e',
                '#e2975d',
                '#e26552',
               
               
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
            // {
            //   label: 'G2',
            //   data: this.state.BarchartdataSecond,
            //   backgroundColor: [
            //     '#3e8e9e', 
            //     '#e9d88f',
            //     '#447c6a',
            //     '#525144',
                
            //   ],
            //   // borderColor: 'red',
            // },
          ],
        }}
        height={200}
        width={300}
        options={{
          title:{
            display:true,
            text:'Some effective Attributes on Grade1'
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
          //onClick:alertBox,
        }}

      />
    </div>
    )
  }
}
