import React, { Component } from 'react';
import { Row,Col,Container,Input} from 'reactstrap';
import Button from '@material-ui/core/Button';
import Header from '../components/Header';
import Footer from './Footer';
import predict from '../assets/images/predict.jpeg';
import AddToQueueIcon from '@material-ui/icons/AddToQueue';
import ArrowBackIcon from '@material-ui/icons/ArrowBack';

import HorizontalLinearStepper1 from './Stepper2';
import axios from 'axios';
import Slider from '@material-ui/core/Slider';
import {Pie} from 'react-chartjs-2';

import { UrlContext } from '../contexts/urlContext';


export default class SecondPrediction extends Component {



  constructor(props) {
    super(props);
    this.state = {

      FirstGrade:'0',
      SecondGrade:'0',
      predictResult:'',
      examFirstGrade:'',
      examSecondGrade:'',
      data:[20, 20],
      'serverUrl': UrlContext._currentValue,    
    };

    // This binding is necessary to make `this` work in the callback
    this.predict = this.predict.bind(this);
  }


  predict(e) {

    e.preventDefault()
    const data = { 
          "G1": this.state.FirstGrade,
          "G2": this.state.SecondGrade
      };
    axios.post(this.state.serverUrl + 'predict/por/G3', data)
    .then(res => {
      const result = res.data;
      // console.log(result)

      this.setState({predictResult:result})
      if (result){
        this.props.history.push({ 
          pathname: '/PredictionResult',
          result:result
         });

      }else{
        alert('Error predicting data!!')
      }

    })
    .catch(()=>{
        alert('Error retrieving data!!');
    })

  

    // console.log(

    //   this.state
    // )


  }

    render() {
        return (

          <Container>
            <Header/>

            <div className="card ">
              <img src={predict} className="img-fluid" alt="Responsive image" style={{"height":"300px","width":"100%"}}></img>
              <div className="card-img-overlay">
                  <h1 className="card-title text-center font-weight-bold text-white" style={{"fontSize":"49px"}} >Start Predict your Final Grade!</h1>
                  
              </div>
            </div>
            <Button className={'mt-3'} onClick={
                  e => {e.preventDefault(); this.props.history.push({pathname: '/Performance'});}} size="large" variant="outlined" color="primary">
                <ArrowBackIcon></ArrowBackIcon><span  className="mr-2">&nbsp;back to Predictions</span> 

              </Button> 
            <div className="container mt-4">
              <HorizontalLinearStepper1  />
            </div>
            <div className="container mt-5 pt-4" style={{ borderBottom:"solid 3px #19738A ","height":"70px"}}>
                <h5 className=" text-center" style={{"top":"40%" , "color": "#19738A "}}  >The most effective Attributes</h5>
              </div>

            <Row className="justify-content-sm-center mt-3">
              <Col className="mt-5 pt-3" md="6">

                <form className="p-3 pr-5">
                  <div className="form-group pt-4">
                    <label for="customRange1" className="form-label my-3">Your first exam grade:</label><br/>

                      <Slider
                        defaultValue={20}
                        // getAriaValueText={valuetext}
                        aria-labelledby="discrete-slider"
                        valueLabelDisplay="auto"
                        step={1}
                        marks
                        min={0}
                        max={20}
                        onChange={(event, newValue) => {
                          this.setState(state => {
                            const data = state.data.map((item, j) => {
                              if (j === 0) {
                                return newValue;
                              } else if(j === 1) {
                                return item;
                              }
                            });
                      
                            return {
                              data,
                            };
                          })
                          
                        }} 
                      />
                  </div>
                  <div className="form-group pt-4">
                    <label for="customRange1" className="form-label my-3">Your second exam grade:</label><br/>

                      <Slider
                        defaultValue={20}
                        // getAriaValueText={valuetext}
                        aria-labelledby="discrete-slider"
                        valueLabelDisplay="auto"
                        step={1}
                        marks
                        min={0}
                        max={20}
                        onChange={(event, newValue) => {
                          this.setState(state => {
                            const data = state.data.map((item, j) => {
                              if (j === 0) {
                                return item;
                              } else if(j === 1) {
                                return newValue;
                              }
                            });
                      
                            return {
                              data,
                            };
                          })
                          
                        }} 
                      />
                  </div>
                </form>
              </Col>


              <Col className="mt-2" md="6" >
                  <Pie
                    data={{
                        labels: ['First Grade', 'Second Grade'],
                        datasets: [
                        {
                            label: 'The Most Effective Attributes On Grade 3',
                            data: this.state.data,
                          backgroundColor: [
                            '#be5168',
                            '#3e8e9e',
                          ],
                      
                          borderWidth: 1,
                        }               
                      
                      ],
                    }}
                    height={200}
                    width={300}
                    options={{
                      title:{
                        display:true,
                        text:'The Most Effective Attributes On Grade 3'
                      },
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
                                   
            </Col>
          </Row>




            <div className="container mt-5 pt-4" style={{ borderBottom :"solid 3px #19738A ","height":"70px"}}>
            <h5 className=" text-center" style={{"top":"40%", "color": "#19738A "}}  >Now try to predict your result</h5>
            </div>

            <Row className="justify-content-sm-center mt-3 ">
            <Col md="6" >

            <div className="card  border-info mt-5   p-3 mb-5 "> 
                 <form>

  <div className="form-row">
  <div className="form-group col-md-6">
      <label>The grade of the first exam:</label>
      <Input
          type="number"
          name="number"
          min="0"
          max="20"
          id="FirstGrade"
          placeholder="0~20"
          onChange={e => {this.setState({'FirstGrade': e.target.value})}}
        />

    </div>
    <div className="form-group col-md-6">
      <label>The grade of the second:</label>
      <Input
          type="number"
          name="number"
          min="0"
          max="20"
          id="SecondGrade"
          placeholder="0~20"
          onChange={e => {this.setState({'SecondGrade': e.target.value})}}
        />

    </div>
  </div>



  <hr className="my-4"/>
                <p className="lead">
                </p>
                <Button onClick={this.predict} size="large" variant="outlined" color="primary">
                <span  className="mr-2">Predict Your Result</span> <AddToQueueIcon></AddToQueueIcon>

</Button>

</form> 
               </div>
            </Col>
            {/* <Col className="mt-5" md="6" >
              <p class="mb-4">The most effective attribute </p>
                <PieChart/>
            </Col> */}
            </Row>

               
            <Footer/>
          </Container>

        )
    }
}
