import React, { Component } from 'react'
import Footer from './Footer'
import { Row,Card, CardTitle, CardText,Col } from 'reactstrap';
import {BrowserRouter as Router, Route} from 'react-router-dom'

import {Link, withRouter ,Switch} from 'react-router-dom';
import Header from './/Header';


import colorfulglobe from '../assets/images/colorfulglobe.png';
import colorfulemail from '../assets/images/colorfulemail.png';
import colorfulmembers from '../assets/images/MMembers.png';
import colorfuldata from '../assets/images/colorfuldataset.png';
import colorfulperformance from '../assets/images/colorfulperformance.png';
import colorfulpiechart from '../assets/images/colorfulpiechart.png';
import introimg from '../assets/images/bgslogan.png';


import data from '../assets/images/data.png';
// import barchart from '../assets/images/Barchart.png';
import globe from '../assets/images/globe.png';
import email from '../assets/images/email.png';
import members from '../assets/images/members.png';
import performance from '../assets/images/performance.png';
import piechart from '../assets/images/piechart.png';











export default class Landing extends Component {
    render() {
        return (
            <Route>
                <div className="container " >
                <Header/>

                    {/* <div className="container-fluid mb-3">

                    </div> */}
                    <Row className="justify-content-sm-start mt-3 ">
                  

                        <Col md="3">
                            <Link to="/Dataset">
                                <Card body inverse id="DatasetBox" style={{ height:'200px',textAlign:'center',backgroundColor:'white', paddingTop: '40px' }}>
                                    <CardText><img src={colorfuldata} alt="Dataset" style={{'border': 'none','height': '60px'}} onMouseOver={e => (e.currentTarget.src =data )} onMouseOut={e => (e.currentTarget.src =colorfuldata )}/></CardText>
                                    <CardTitle tag="h6" style={{"color":"#B06660"}}>Dataset</CardTitle>
                                    {/* <Button>Button</Button> */}
                                </Card>
                            </Link>
                        </Col> 
                        <Col md="3">
                            <Link to="/DecisionTreeMainPage">
                                <Card body inverse  id="VisualizationBox" style={{ height:'200px',backgroundColor:'white' ,textAlign:'center', paddingTop: '40px'}}>
                                    <CardText><img src={colorfulpiechart} alt="Decision Tree" style={{'border': 'none','height': '60px'}} onMouseOver={e => (e.currentTarget.src =piechart )} onMouseOut={e => (e.currentTarget.src =colorfulpiechart )}/></CardText>
                                    <CardTitle tag="h6" style={{"color":"#6A7D8E"}}>Decision Tree</CardTitle>
                                    {/* <Button color="secondary">Button</Button> */}
                                </Card>
                            </Link>
                        </Col> 
                        <Col md="3">
                            <Link to="/Performance">
                                <Card body inverse  id="PerformanceBox" style={{ height:'200px',backgroundColor:'white', textAlign:'center', paddingTop: '40px'}}>
                                    <CardText><img src={colorfulperformance} alt="Performance" style={{'border': 'none','height': '60px'}} onMouseOver={e => (e.currentTarget.src =performance )} onMouseOut={e => (e.currentTarget.src =colorfulperformance )}/></CardText>
                                    <CardTitle tag="h6" style={{"color":"#AB9C73"}}>Performance</CardTitle>

                                    {/* <Button color="secondary">Button</Button> */}
                                </Card>
                            </Link>
                        </Col> 
                    </Row> 
                    

                    
                    <Row className="justify-content-sm-start mt-4 ">
                        <Col md="3">
                            <a href="https://www.uni-due.de/soco/" target="_blank">
                                <Card body inverse id="socoBox" style={{ height:'200px',textAlign:'center',backgroundColor:'white', paddingTop: '40px'}}>
                                    <CardText><img src={colorfulglobe} alt="socoBox" style={{'border': 'none','height': '60px' }} onMouseOver={e => (e.currentTarget.src =globe )} onMouseOut={e => (e.currentTarget.src =colorfulglobe )}/></CardText>
                                    <CardTitle tag="h6" style={{"color":"#587992"}}>Social Computing Group</CardTitle>
                                    {/* <Button color="secondary">Button</Button> */}
                                </Card>
                            </a>
                        </Col> 
                        <Col md="3" style={{'zIndex':' 0'}}>
                            <Link to="/AboutUs">
                                <Card body inverse  id="AboutUsBox"  style={{ height:'200px',textAlign:'center',backgroundColor:'white' , paddingTop: '40px'}}>
                                <CardText><img src={colorfulmembers} alt="AboutUsBox" style={{'border': 'none','height': '60px'}} onMouseOver={e => (e.currentTarget.src =members )} onMouseOut={e => (e.currentTarget.src =colorfulmembers )}/></CardText>
                                    <CardTitle tag="h6" style={{"color":"#74A18E"}}>About us</CardTitle>

                                    {/* <Button color="secondary">Button</Button> */}
                                </Card>
                            </Link>
                        </Col> 
                    </Row>  

                    <Row className="justify-content-sm-start mt-4 ">
                        <Col md="3">
                            <Link to="/ContactUs">
                                <Card body inverse  id="ContactUsBox"  style={{ height:'200px',textAlign:'center',backgroundColor:'white', paddingTop: '40px'}}>
                                <CardText><img src={colorfulemail} alt="ContactUs" style={{'border': 'none','height': '60px'}} onMouseOver={e => (e.currentTarget.src =email )} onMouseOut={e => (e.currentTarget.src =colorfulemail )}/></CardText>
                                    <CardTitle tag="h6" style={{"color":"#CA8F42"}}>Contact us</CardTitle>

                                    {/* <Button color="secondary">Button</Button> */}
                                </Card>
                            </Link>
                        </Col> 
                        <Col md="8" >
                        <img src={introimg} alt="img" style={{'border': 'none'  , marginLeft:'250px', marginTop:'130px' ,'opacity':' 0.9'}}/>
                        </Col>
                    </Row>  


                    <br></br>
                    <Footer />
                </div>
            </Route>
        )
    }
}
