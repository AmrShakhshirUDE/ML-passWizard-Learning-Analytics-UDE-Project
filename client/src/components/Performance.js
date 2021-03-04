import React, { Component } from 'react'
import Footer from './Footer'
import { Row,Card, CardTitle, CardText,Col } from 'reactstrap';
import {BrowserRouter as Router, Route} from 'react-router-dom'
import Header from './/Header';
import {Link, withRouter ,Switch} from 'react-router-dom';
import machinelearning from '../assets/images/MachineLearning.jpeg'
import ML from '../assets/images/ML.jpeg'
import data from '../assets/images/data.png';

import firststep from '../assets/images/firststep.png';
import secondstep from '../assets/images/secondstep.png';
import thirdstep from '../assets/images/thirdstep.png';
import ReactTooltip from 'react-tooltip';
import HorizontalLinearStepper from './Stepper'




export default class Performance extends Component {
    render() {
        return (
                <div className="container">
                    <Row className="justify-content-sm-center mt-3 ">
                <Header/>

                    <div className="card  text-white" >
                      <img src={machinelearning} className="img-fluid" alt="Responsive image" style={{"height":"300px","width":"100%"}}></img>
                      <div className="card-img-overlay">
                          <h1 className="card-title text-center font-weight-bold" style={{"fontSize":"49px"}} >Machine Learning</h1>
                          
                      </div>
                    </div> 

        <div className="container my-3">
          <HorizontalLinearStepper />
          </div>

          <Row className="justify-content-sm-center my-4 ">
            <Col md="4" >
                    <Link to="/FirstPrediction">
                        <Card body inverse id="ContactUsBox" style={{border: '1px solid #cacaca' , width: '100%' ,height:'200px',textAlign:'center',backgroundColor:'white', paddingTop: '40px' }}>
                            <CardText><img src={firststep} alt="Performance" style={{'border': 'none','height': '80px'}}></img></CardText>
                            <CardTitle tag="h6" style={{color:"#7c5042"}}>I am at the <b>begining</b> of the semester.</CardTitle>
                        </Card>
                    </Link>
                </Col> 
                <Col md="4">
                    <Link to="/SecondPrediction">
                        <Card body inverse id="VisualizationBox" style={{border: '1px solid #cacaca' ,width: '100%', height:'200px',backgroundColor:'white' ,textAlign:'center', paddingTop: '40px'}}>
                            <CardText><img src={secondstep} alt="Performance" style={{'border': 'none','height': '80px'}}></img></CardText>
                            <CardTitle tag="h6" style={{"color":"#4454b4"}}>I have done <b>one</b> exam.</CardTitle>
                        </Card>
                    </Link>
                </Col> 
                <Col md="4">
                    <Link to="/ThirdPrediction">
                        <Card body inverse id="AboutUsBox" style={{border: '1px solid #cacaca' ,width: '100%', height:'200px',backgroundColor:'white', textAlign:'center', paddingTop: '40px'}}>
                            <CardText><img src={thirdstep} alt="Performance" style={{'border': 'none','height': '80px'}}></img></CardText>
                            <CardTitle tag="h6" style={{"color":"#53633a"}}>I have done <b>two</b> exams</CardTitle>
                        </Card>
                    </Link>
                </Col> 
            </Row>


                      <div  className="container mt-5" style={{"backgroundColor":"#f5f5f1"}}>
                    <div className="container mt-5"  >
                    <div>
              <h1 className="my-5 text-center" data-tip="You can choose three options above to predict your result!" >WHICH STATAGE ARE YOU AT?</h1>
              <ReactTooltip className='extraClass' effect="solid" width="180px" height="100px" />
            </div>
            <div className="col-sm-12 " style={{marginLeft:"-15px"}} >
                      <div className="" style={{"backgroundColor":"#f5f5f1"}}>
                        <div className="card-body">
                          <p align="justify ">
                    To get your exam's status predicted, you have three choices:
                    <li><strong style={{color:"#34666F"}}>1 :</strong> At the beginning of a semester (you haven't been examined yet):<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output: weather you will pass or fail at each exam (G1, G2, G3).</li>
                    <li> <strong style={{color:"#34666F"}}>2 :</strong>After completing the first exam (G1):<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output: your result at second exam (G2): fail, medium [10 to less than 15], high [15 to 20].</li>
                    <li> <strong style={{color:"#34666F"}}>3 :</strong>After completing the second exam (G2):<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output: your result at final exam (G3): fail, medium [10 to less than 15], high [15 to 20].</li>

                    </p>
                        
                        </div>
                      </div>
                    </div>
                
                

                    </div>
                      <div   className="d-flex mb-3">
                    <div className="col-sm-6" >
                      <div className="card border-info" style={{"backgroundColor":"#f5f5f1"}}>
                        <div className="card-body">
                          <h5 className="card-title font-weight-bold" style={{"color":"#34666F"}}>Workflow</h5>
                          <p className="card-text justify">
                          In order to get results as accurate as possible, several techniques and algorithms have been applied such as deriving new attributes, selecting most effective features based on different algorithms of feature selection, and clustering some ordinal values into groups. The resulted dataset has been used by to train different machine learning algorithms by Scikit-learn. By changing train/ test ratio and comparing the different models, the model with the highest accuracy value has been chosen to be used in this application.
                         </p>
                        
                        </div>
                      </div>
                    </div>

                    <div className="col-sm-6 mt-1 mb-2">
                      <div className="card border-info" style={{"backgroundColor":"#f5f5f1"}}>
                        <div className="card-body">
                          <h5 className="card-title font-weight-bold" style={{"color":"#34666F"}}>Results</h5>
                          <p className="card-text justify ">
                          As a result, the used algorithm for prediction models is support vector classification (SVC).
The used ratio for train/test data is 80/20 respectively.
The accuracy value is based on precision value, that is, the fraction of relevant instances among the retrieved instances.
For more understanding of features effects, please try the interactive decision tree here and be aware that the accuracy in decision tree models is less than the accuracy in SVC models that are mainly used in this application
                           </p>
                        </div>
                      </div>         
                    </div>
                    </div>
                    </div>
                   
                                     
              </Row>

      

  
              <br></br>
              <Footer />
          </div>
        )
    }
}
