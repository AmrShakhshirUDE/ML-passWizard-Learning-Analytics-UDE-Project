import React, { Component } from 'react'
import Footer from './Footer'
import { Row,Col} from 'reactstrap';


import location from '../assets/images/location.png';
import googleplus from '../assets/images/googleplus.png';
import whatsapp from '../assets/images/whatsapp.png';
import twitter from '../assets/images/twitter.png';
import email2 from '../assets/images/email2.png';
import phone2 from '../assets/images/phone2.png';
import rss from '../assets/images/rss.png';
import contact from '../assets/images/contact.jpg';
import Header from './Header'
import Button from '@material-ui/core/Button';









export default class ContactUs extends Component {
    render() {
        return (
                <div className="container" >
                    <Header />
                      <div className="card  ">
                        <img src={contact} className="img-fluid" alt="Responsive image" style={{"height":"450px","width":"100%"}}></img>
                        <div className="card-img-overlay">
                            <h1 className="card-title text-center font-weight-bold text-black" style={{"fontSize":"49px"}} ></h1>
                           
                        </div>
                        </div>
                    <Row className=" mt-3 ">
                        <Col md="12" className="text-center font-weight-bold mt-5">    
                       <h3> CONTACT US</h3>
                        </Col>

                    </Row>
                    <br/><br/><br/>
                    <Row className="justify-content-center mt-3 ">

                        <Col md="5" className="justify-content-center ">
                        <div className="card  border-info p-3 mb-5  rounded "> 
                 <form>
                <div className="form-row">
                    
                
                </div>
                <div className="form-group">
                    <label for="inputAddress"> <i className="fa fa-envelope  text-info mr-1"></i> Email</label>
                    <input type="text" className="form-control" id="inputAddress" placeholder="abc@gmail.com" />
                </div>
                <div className="form-group">
                    <label for="inputAddress2"> <i className="fa fa-user text-info mr-1"></i>Name</label>
                    <input type="text" className="form-control" id="inputAddress2" placeholder="your name"/>
                </div>
                <div className="form-group">
                    <label for="exampleFormControlTextarea1"> <i className="fa fa-comment text-info mr-1"></i>Message</label>
                    <textarea className="form-control" id="exampleFormControlTextarea1" rows="3"></textarea>
                </div>
                <hr className="my-4"/>
                <p className="lead">
                  {/* <a className="btn btn-primary btn-lg" href="#" role="button">Predict Your Result</a> */}
                </p>
                <Button onClick={this.predict} size="large" variant="outlined" color="primary">
                <span  className="mr-2">Submit</span> 

</Button>

</form> 
</div>

                        </Col>
                        <Col md="1"></Col>
                        <Col md="5">
                        <br/>
                            <Row className="justify-content-center mt-5 ">
                                <Col md="6" className="justify-content-center text-center">
                                    <img src={location} alt="location" style={{'border': 'none','height': '25px'}}/>   
                                </Col>
                                <Col md="6" className="justify-content-center ">
                                Universität Duisburg-Essen
                                Fakultät für Ingenieurwissenschaften
                                Abteilung INKO
                                Fachgebiet Social Computing
                                47048 Duisburg
                                </Col>

                            </Row>
                            <Row className="justify-content-center mt-3">
                            <Col md="6" className="justify-content-center   text-center">
                                    <img src={email2} alt="email2" style={{'border': 'none','height': '25px'}}/>   
                                </Col>
                                <Col md="6" className="justify-content-center ">
                                    lapro@gmail.com
                                </Col>

                            </Row>
                            <Row className="justify-content-center mt-3">
                            <Col md="6" className="justify-content-center  text-center ">
                                    <img src={phone2} alt="phone2" style={{'border': 'none','height': '25px'}}/>   
                                </Col>
                                <Col md="6" className="justify-content-center ">
                                    +12345678910
                                </Col>

                            </Row>
                            <br/><br/><br/>
                            <Row className="text-center mt-3 border-top border-secondary py-3">
                                <Col md="3" className="text-center">
                                    <img src={twitter} alt="twitter" style={{'border': 'none','height': '25px'}}/>   
                                </Col>
                                <Col md="3" className="text-center ">
                                    <img src={whatsapp} alt="whatsapp" style={{'border': 'none','height': '25px'}}/>   
                                </Col>
                                <Col md="3" className="text-center">
                                    <img src={googleplus} alt="googleplus" style={{'border': 'none','height': '25px'}}/>   
                                </Col>
                                <Col md="3" className="text-center">
                                    <img src={rss} alt="rss" style={{'border': 'none','height': '25px'}}/>   
                                </Col>
                            </Row>
                                
                        </Col>

                        
                    </Row>
                    <br></br>
                    <Footer />
                </div>
        )
    }
}
