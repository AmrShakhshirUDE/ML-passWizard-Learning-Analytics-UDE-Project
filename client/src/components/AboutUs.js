import React, { Component } from 'react'
import Footer from './Footer'
import { Row } from 'reactstrap';
import Header from './/Header';
import teamwork from '../assets/images/aboutus.jpg';
import GroupIcon from '@material-ui/icons/Group';
import HomeWorkIcon from '@material-ui/icons/HomeWork';
import LocationCityIcon from '@material-ui/icons/LocationCity';








export default class AboutUs extends Component {
    render() {
        return (
                <div className="container">
                    <Row className="justify-content-sm-start mt-3 ">
            <Header/>
           
            <div className="card  text-white">
                        <img src={teamwork} className="img-fluid" alt="Responsive image" style={{"height":"300px","width":"100%"}}></img>
                        <div className="card-img-overlay">
                            <h1 className="card-title text-center font-weight-bold" style={{"fontSize":"49px","color":"white"}} >Our Team</h1>
                           
                        </div>
                        </div> 

         
                <div className="">
                <div className="card-body my-5">
                    <h5 className="card-title ">Team Members</h5>
                    <p className="card-text ">With supporting text be
                    ng text below as a natural leng text below as a natural le
                    low as a natural lead-in 
                    With supporting text be ng text below as a natural leng text below as a natural le low as a natural lead-in ng text below as a natural le ng text below as a natural leto additional content.
                    With supporting text be ng text below as a natural leng text below as a natural le low as a natural lead-in ng text below as a natural le ng text below as a natural leto additional content.
                    ng text below as a natural le
                    ng text below as a natural leto additional content.</p>
                </div>
                </div>

                       
            <div className="d-flex container justify-content-center1" style={{"height":"200px"}}>
                    <div className=" mx-5 mt-3" style={{"padding-top":"30px"}}>
                      <GroupIcon  style={{ fontSize: 65,fill: "white" }}></GroupIcon>
                    {/* <img src={aboutus} height="65px"/> */}
                    <p className="ptitle text-white">WHO WE ARE</p>
                        </div>  
                    <div className=" mx-5 mt-3" style={{"padding-top":"30px"}}>
                      <LocationCityIcon  style={{  fontSize: 65,fill: "white"  }}></LocationCityIcon>
                    {/* <img src={member} height="65px"/> */}
                    <p className="ptitle text-white">WHERE WE AT</p>
                        </div>  
                    <div className="mx-5 mt-3" style={{"padding-top":"30px"}}>
                      <HomeWorkIcon   style={{  fontSize: 65,fill: "white"  }}></HomeWorkIcon>
                    {/* <img src={aboutus} height="65px"/> */}
                    <p className="ptitle text-white">WHAT WE CAN DO</p>
                        </div>  
            </div> 

    

  <div className="container mt-5 ">
    <div className="boxes d-flex">
    <div className="card border-info text-center mx-2 col-sm-6">
    <div className="card-body">
      <div className="box ">
        <img className="icon" src="https://img.icons8.com/plasticine/100/000000/html.png"/>
      <h3><span className="text-primary">Frontend</span></h3>

      <div className="feature">
         HTML / CSS / Javascript
      </div>
      <div className="feature">
        Angular
      </div>
        <div className="feature">
        React
      </div>
        <div className="feature">
         Bootstrap
      </div>
   
      <div className="feature">
         Wordpress
      </div>
      <div className="feature">
         Git version control
      </div>
      <div className="feature">
        Responsive layout and design
      </div>
  </div>
  </div>
    </div>
    <div className="card border-info text-center col-sm-6 mx-2">
    <div className="card-body">
    <div className="box">
          <img className="icon" style={{"max-width":"100%","height":"auto"}} src="https://img.icons8.com/bubbles/100/000000/database.png"/>
        <h3><span className="text-primary ">Backend</span></h3>
        <div className="feature">
         Python
        </div>
          <div className="feature">
        Flask
        </div>
          <div className="feature">
         NodeJS
        </div>
        <div className="feature">
     C++
        </div>
        <div className="feature">
     MySQL
        </div>
        <div className="feature">
       MongoDB
        </div>
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
