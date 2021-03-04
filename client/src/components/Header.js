import React, { Component } from 'react';
import { Row,Col,Container} from 'reactstrap';
import logo from '../assets/images/Blacklogo.png';
import goal from '../assets/images/goal.png'
export default class Header extends Component {
    render() {
        return (

          <Container fluid>
<nav className="navbar navbar-expand-lg navbar-light" style={{"height":"100px"}}>
  <div className="container-fluid justify-content-between">
    <div className="d-flex">
    

      
       
    </div>

    <ul className="navbar-nav flex-row d-none d-md-flex">
      <li className="nav-item me-3 me-lg-1 active">
      
    
      </li>

      <li className="nav-item me-3 me-lg-1">
      </li>

      <li className="nav-item me-3 me-lg-1">
      
      </li>

      <li className="nav-item me-3 me-lg-1">
      
      </li>

      <li className="nav-item me-3 me-lg-1">
     
      </li>
    </ul>

    <ul className="navbar-nav flex-row">
      <li className="nav-item me-3 me-lg-1">
        <a className="nav-link d-sm-flex align-items-sm-center" href="#">
        <img
          src={goal}
          height="30"
          alt=""
          loading="lazy"
          style={{"marginTop": "0px"}}
        />
             <a className="nav-link" href="https://flaskpro-advwebtech.herokuapp.com/">

          <strong className="d-none d-sm-block ms-1">
            Last Project</strong></a>
        </a>
      </li>
      <li className="nav-item me-3 me-lg-1">
        <a className="nav-link" href="#">
          <span>        <i className="fa fa-github  mt-2" aria-hidden="true" style={{"fontSize":"25px","color":"#ba7d9d"}}></i>
</span>
        </a>
      </li>
      <li className="nav-item  me-3 me-lg-1">
      <a className="nav-link" href="#">
          <span>        <i className="fa fa-linkedin mt-2 " aria-hidden="true" style={{"fontSize":"25px","color":"#a0b9ba"}}></i>
</span>
        </a>
      </li>
      <li className="nav-item  me-3 me-lg-1">
      <a className="nav-link" href="#">
          <span> 
            <i className="fa fa-xing mt-2" aria-hidden="true" style={{"fontSize":"25px","color":"rgba(233,216,141)"}}></i>
</span>
        </a>
      </li>
   
     
    </ul>
  </div>
</nav>
          </Container>
          

        )
    }
}
