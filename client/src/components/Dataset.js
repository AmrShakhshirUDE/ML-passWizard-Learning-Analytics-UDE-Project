import React, { Component } from 'react'
import Footer from './Footer'
import { Row,Card, CardTitle, CardText,Col } from 'reactstrap';
import {BrowserRouter as Router, Route} from 'react-router-dom'
import dataViz from '../assets/images/dataViz.png';
import {Link, withRouter ,Switch} from 'react-router-dom';
import datamind from '../assets/images/DataMind.jpeg';
import Header from './/Header';
import PermDataSettingIcon from '@material-ui/icons/PermDataSetting';
import globe from '../assets/images/globe.png';
import email from '../assets/images/email.png';
import members from '../assets/images/members.png';
import barchart from '../assets/images/Barchart.png';
import data from '../assets/images/data.png';
import performance from '../assets/images/performance.png';
import AssessmentIcon from '@material-ui/icons/Assessment';
import Visualization from './Visualization'





export default class Dataset extends Component {
    render() {
        return (
                <div className="container">
                    <Row className="justify-content-sm-start mt-3 ">
                    <Header/>
                        {/* <img src={datamind} class="img-fluid" alt="Responsive image"></img> */}
          
                        <div className="card  text-white">
                        <img src={datamind} className="img-fluid" alt="Responsive image" style={{"height":"300px","width":"100%"}}></img>
                        <div className="card-img-overlay">
                            <h1 className="card-title text-center font-weight-bold" style={{"fontSize":"49px"}} >Meet Our Dataset</h1>
                           
                        </div>
                        </div>

                        {/* <img src={datamind} class="img-fluid" alt="Responsive image" style={{"height":"450px","width":"100%"}}></img> */}
 
                        <div className="container" style={{}}>
                            <div className="container mt-5 mb-3">
                            <h3> DataSet Description</h3>
                            </div>
                            {/* <div class="row"> */}
                    <div className="container my-5">
                        <div className="card border-info">
                        <div className="card-header font-weight-bold h5" style={{"color":"#34666F","backgroundColor":"#B2C0C1"}}>
                        <PermDataSettingIcon></PermDataSettingIcon>Multivariate dataset                    </div>
                    <div className="card-body">
                        <p className="card-text">
                        <li>Represents students' performance in Portuguese course.</li>
                <li>Number of students: 649</li>
                <li>Number of attributes: 30</li>
                <li>[Include student grades, demographic, social and school related features]</li> 
                <li>Target attributes: 3</li>   
                <li>[Show students' grades in first, second, and final period exams (G1, G2, G3) respectively]</li>
                <li>For more information on dataset please visit <a href="https://archive.ics.uci.edu/ml/datasets/student+performance#">this</a> page.</li>                         
                </p>
                        </div>
                    </div>
                </div>
                <div className="container mt-5">
                    <div className="card border-info">
                    <div className="card-header font-weight-bold h5"style={{"color":"#34666F","backgroundColor":"#B2C0C1"}} >
                   <AssessmentIcon></AssessmentIcon> Attributes of dataset                    </div>
                    <div className="card-body">
 
                        <p className="card-text">
                            <li> There 33 attributes in total [30 features + 3 target]. Below you can find a set of the most important ones:</li>
                        <li><strong style={{color:"#34666F"}}>School</strong>: student's school ('GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)</li>
                    <li><strong style={{color:"#34666F"}}>Sex</strong>: student's sex (female or male)</li>
                    <li><strong style={{color:"#34666F"}}>Medu</strong>: mother's education (none, primary education (4th grade), 5th to 9th grade, secondary education or higher education)</li>
                    <li><strong style={{color:"#34666F"}}>Fedu</strong>: father's education (none, primary education (4th grade), 5th to 9th grade, secondary education or higher education)</li>
                    <li><strong style={{color:"#34666F"}}>Mjob</strong>: mother's job ('teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')</li>
                    <li><strong style={{color:"#34666F"}}>Traveltime</strong>: home to school travel time (1 - more than 15 min., 15 to 30 min., 30 min. to 1 hour, or less than 1 hour)</li>
                    <li><strong style={{color:"#34666F"}}>Studytime</strong>: weekly study time (less than 2 hours, 2 to 5 hours, 5 to 10 hours, or more than 10 hours)</li>
                    <li><strong style={{color:"#34666F"}}>Failures</strong>: number of past class failures (1, 2, 3, or more than three)</li>
                    <li><strong style={{color:"#34666F"}}>Higher </strong>: wants to take higher education (yes or no)</li>
                    <li><strong style={{color:"#34666F"}}>Freetime</strong>: free time after school (very low, low, neutral, high, very high)</li>
                    <li><strong style={{color:"#34666F"}}>Dalc </strong>: workday alcohol consumption (very low, low, neutral, high, very high)</li>
                    <li><strong style={{color:"#34666F"}}>Walc</strong>: weekend alcohol consumption (very low, low, neutral, high, very high)</li>   
                    <li><strong style={{color:"#34666F"}}>Absences</strong>: number of school absences (0 ~ 32)</li>   
                    <li><strong style={{color:"#34666F"}}>G1</strong>: first period grade (from 0 to 20)</li>   
                    <li><strong style={{color:"#34666F"}}>G2</strong>: second period grade (from 0 to 20)</li>   
                    <li><strong style={{color:"#34666F"}}>G3</strong>: final grade (0 to 20)</li>   
                    <li>For more information on other attributes please visit <a href="https://archive.ics.uci.edu/ml/datasets/student+performance">this</a> page.</li>                         


                     </p>
                    </div>
                    </div>
                </div>
             
                
                <div className="mt-4">
                    <Visualization/>
                 <Link to="/Performance" style={{ textDecoration: 'none' }}>
                          <Card body inverse id="DatasetBox"  style={{ height:'200px',textAlign:'center',backgroundColor:'white', paddingTop: '40px' }}>
                              <CardText><img src={dataViz} alt="Performance" style={{'border': 'none','height': '100px'}}></img></CardText>
                              <CardTitle tag="h5"  style={{"color":"#19738A","font-weight":"bold"}}>Click <b>to</b> predict! </CardTitle>
                          </Card>
                      </Link>
                      </div>

                </div>
                {/* </div> */}
                
                
                

       
                       
                    </Row>
                    <br></br>
                    <Footer />
                </div>
        )
    }
}
