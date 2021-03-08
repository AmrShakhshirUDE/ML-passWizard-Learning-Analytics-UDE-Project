import React, { Component } from 'react';
import { Row,Col,Container,Input,CustomInput} from 'reactstrap';
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

      higher:'0',
      absences:'0',
      failures:'0',
      WeekendAlcohol:'0',
      MotherEducational:'0',
      FatherEducational:'0',
      FirstGrade:'0',
      predictResult:''    ,
      examFirstGrade:''  ,
      examFailures:'',
      examHigher:'',
      data:[20, 15, 0],
      'serverUrl': UrlContext._currentValue,
    };

    // This binding is necessary to make `this` work in the callback
    this.predict = this.predict.bind(this);

  }

  predict(e) {

    e.preventDefault()
    const data = { 
      "higher": this.state.higher,
      "absences": this.state.absences,
      "failures": this.state.failures,
      "Walc": this.state.WeekendAlcohol,
      "Medu": this.state.MotherEducational,
      "Fedu": this.state.FatherEducational,
      "G1": this.state.FirstGrade
    };
    axios.post(this.state.serverUrl + `predict/por/G2`, data)
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

  }



    render() {

        return (



          <Container>
            <Header/>

              <div className="card  ">
                <img src={predict} className="img-fluid" alt="Responsive image" style={{"height":"300px","width":"100%"}}></img>
                  <div className="card-img-overlay">
                    <h1 className="card-title text-center font-weight-bold text-white" style={{"fontSize":"49px"}} >Start Predict your Second Grade!</h1>  
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
              <Row className="justify-content-sm-center mt-3 ">
              <Col className="mt-5" md="6">

                <form className="p-3 pr-5">
                  <div className="form-group pt-4 ">
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
                              }else{
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
                  <div className="form-group my-3">
                      <label>Number of exam failures:</label>
                      <Input
                          type="number"
                          name="examFailures"
                          min="0"
                          max="20"
                          id="FirstGrade"
                          placeholder="0~20"
                          onChange={e => {
                            this.setState(state => {
                              const data = state.data.map((item, j) => {
                                if (j === 0) {
                                  return item;
                                } else if(j === 1) {
                                  return e.target.value;
                                }else{
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
                  <div className="form-check my-5 pl-0">
                    
                  <CustomInput type="switch" id="exampleCustomSwitch" name="customSwitch" label="Are you willing to take higher education?"
                  
                  onChange={e => {
                    if (e.target.checked){

                      this.setState(state => {
                        const data = state.data.map((item, j) => {
                          if (j === 0) {
                            return item;
                          } else if(j === 1) {
                            return item;
                          }else{
                            return 10;
                          }
                        });
                  
                        return {
                          data,
                        };
                      })
                      
                      }else{
                        this.setState(state => {
                          const data = state.data.map((item, j) => {
                            if (j === 0) {
                              return item;
                            } else if(j === 1) {
                              return item;
                            }else{
                              return 0;
                            }
                          });
                    
                          return {
                            data,
                          };
                        })
                      }
                    } 
                    }
                    
                  />    

                  </div>

                </form>
              </Col>


              <Col className="mt-2" md="6" >
                  <Pie
                    data={{
                        labels: ['First Grade', 'Exam Failures', 'Higher Education'],
                        datasets: [
                        {
                            label: 'The Most Effective Attributes On Grade 2',
                            data: this.state.data,
                          backgroundColor: [
                            '#be5168',
                            '#3e8e9e',
                            '#e2975d',
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
                        text:'The Most Effective Attributes On Grade 2'
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

            <div className="card  border-info mt-5   p-3 mb-5"> 
                 <form>
  <div className="form-row">
    <div className="form-group col-md-12">
      <label>Do you want to take higher education:</label>
      <Input type="select" name="select" id="higher" onChange={e => {this.setState({'higher': e.target.value})}}>
            <option value="0">Yes</option>
            <option value="1">No</option>
      </Input>
    </div>
  </div>
  <div className="form-row">
    <div className="form-group col-md-6">
      <label>Number of school absences:</label>
      <Input
          type="number"
          name="number"
          min="0"
          max="32"
          id="absences"
          placeholder="0~32"
          onChange={e => {this.setState({'absences': e.target.value})}}
        />
    </div>
    <div className="form-group col-md-6">
      <label>Number of exam failures:</label>
      <Input type="select" name="select" id="failures" onChange={e => {this.setState({'failures': e.target.value})}}>
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">else</option>
      </Input>
    </div>
  </div>

  <div className="form-row">      
    <div className="form-group col-md-12">
      <label>Weekend alcohol consumption:</label>
      <Input type="select" name="select" id="WeekendAlcohol" onChange={e => {this.setState({'WeekendAlcohol': e.target.value})}}>
            <option value="0">very low</option>
            <option value="1">low</option>
            <option value="2">neutral</option>
            <option value="3">high</option>
            <option value="4">very high</option>
      </Input>
    </div>
  </div>

  <div className="form-row">      
    <div className="form-group col-md-6">
      <label>Mother educational background:</label>
      <Input type="select" name="select" id="MotherEducational" onChange={e => {this.setState({'MotherEducational': e.target.value})}}>
            <option value="0">none</option>
            <option value="1">primary education (4th grade)</option>
            <option value="2">5th to 9th grade</option>
            <option value="3">secondary education</option>
            <option value="4">higher education</option>
      </Input>
    </div>
    <div className="form-group col-md-6">
      <label>Father educational background:</label>
      <Input type="select" name="select" id="FatherrEducational" onChange={e => {this.setState({'FatherrEducational': e.target.value})}}>
            <option value="0">none</option>
            <option value="1">primary education (4th grade)</option>
            <option value="2">5th to 9th grade</option>
            <option value="3">secondary education</option>
            <option value="4">higher education</option>
      </Input>
    </div>
  </div>
  <div className="form-row">      
    <div className="form-group col-md-12"> 
      <label >The grade of your first exam:</label>
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
  </div>


  <hr className="my-4"/>
                <p className="lead">
                  {/* <a className="btn btn-primary btn-lg" href="#" role="button">Predict Your Result</a> */}
                </p>
                <Button onClick={this.predict} size="large" variant="outlined" color="primary">
                <span  className="mr-2">Predict Your Result</span> <AddToQueueIcon></AddToQueueIcon>

</Button>

</form> 
               </div>

            </Col>
            </Row>

         


               
            <Footer/>
          </Container>

        )
    }
}
