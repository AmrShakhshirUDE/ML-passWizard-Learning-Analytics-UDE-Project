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
                    <h5 className="card-title "  style={{"color":"#34666F"}}>LAPro Group</h5>
                    <p className="card-text ">We are LAPro group which is a group of 4 master students from all over the word who wants to success in his/her career and gain more experiences in computer science. Each of us has their own strengths. Baohui likes to design and searching in internet. She makes us always uptodated. Amr likes programming and likes to spend much time on machine learning algorithm and modify the attributes and get different results. Tannaz loves designing and coding, she looks always toward an userfriendly interface. Hesam is Network Profi and will solve all the problems with deploying and connections. 
                    
                     </p>
                     <h6  style={{"color":"#34666F"}}>
                     We work together with great passion and perseverance.
                    We offer with talents in the following fields: Web design, back-end development, front-end development.</h6>
                </div>
                </div>
                       
            <div className="d-flex container justify-content-center1" style={{"height":"350px"}}>
                    <div className=" mx-5 mt-3" style={{"padding-top":"30px"}}>
                      <GroupIcon  style={{ fontSize: 65,fill: "white" }}></GroupIcon>
                    {/* <img src={aboutus} height="65px"/> */}
                    <p className="ptitle text-white">WHO WE ARE</p>

                    <p className="ptitle text-white">Baohui Deng</p>
                    <p className="ptitle text-white">Tannaz Vahidi</p>
                    <p className="ptitle text-white">Amr Shakhshir</p>
                    <p className="ptitle text-white">Hesam Heidarzadeh</p>



                        </div>  
                    <div className=" mx-5 mt-3" style={{"padding-top":"30px"}}>
                      <LocationCityIcon  style={{  fontSize: 65,fill: "white"  }}></LocationCityIcon>
                    {/* <img src={member} height="65px"/> */}
                    <p className="ptitle text-white">WHERE WE ARE</p>
                    <p className="ptitle text-white">Duisburg-Essen University</p>
                    <p className="ptitle text-white">Germany</p>

                        </div>  
                    <div className="mx-5 mt-3" style={{"padding-top":"30px"}}>
                      <HomeWorkIcon   style={{  fontSize: 65,fill: "white"  }}></HomeWorkIcon>
                    {/* <img src={aboutus} height="65px"/> */}
                    <p className="ptitle text-white">WHAT WE CAN DO</p>
                    <p className="ptitle text-white">Programming</p>

                        </div>  
            </div> 

    

  <div className="container mt-5 ">
    <div className="boxes d-flex">
    <div className="card border-info text-center mx-1 col-sm-6">
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
        React.js
      </div>
        <div className="feature">
         Bootstrap
      </div>
   
      <div className="feature">
         Wordpress
      </div>

  </div>
  </div>
    </div>
    <div className="card border-info text-center col-sm-6 mr-4">
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
     PHP
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
