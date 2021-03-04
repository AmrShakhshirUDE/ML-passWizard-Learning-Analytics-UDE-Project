import React from 'react';
import {BrowserRouter as Router, Route} from 'react-router-dom'
import { Row,Col,Container} from 'reactstrap';

import {Switch} from 'react-router-dom';

import MetisMenu from 'react-metismenu';

import AboutUs from './components/AboutUs';
import ContactUs from './components/ContactUs';
import Dataset from './components/Dataset';
import Performance from './components/Performance';
import Visualization from './components/Visualization';
import Landing from './components/Landing';
import NotFound from './components/NotFound';

import logo from './assets/images/whitelogo.png';
import FirstPrediction from './components/FirstPrediction';
import SecondPrediction from './components/SecondPrediction';
import ThirdPrediction from './components/ThirdPrediction';
import Result from './components/Result'

import DecisionTreeMainPage from './components/DecisionTreeMainPage';



const menu = [
    {
      label: <img src={logo} alt="Logo" style={{'border': 'none','height': '100px', 'paddingTop': '20px','paddingBottom': '20px'}}/>
    },
    {
      icon: '',
      label: 'Home',
      to: '/',
    },
    {
      icon: '',
      label: 'Data',
      content: [
        {
          icon: '',
          label: 'Dataset',
          to: '/Dataset',
        },
        {
          icon: '',
          label: 'Decision Tree',
          to: '/DecisionTreeMainPage',
        }
      ],
    },
    {
      icon: '',
      label: 'Prediction',
      to: '/Performance',
    },
    ,
    {
      icon: '',
      label: 'Social Computing Group',
      externalLink: true,
      to: 'https://www.uni-due.de/soco/',
    },
  ];


function App() {
  return (
    <div className="App">
      <Router>
      

        <Container fluid>
              <Row>
                  <Col md="2">
                      <MetisMenu
                      content={menu}
                      activeLinkFromLocation
                      />
                  </Col>
                  <Col md="10"  >

                    <Switch>
                      <Route exact path="/" component = {Landing} />
                      <Route exact path="/Dataset" component = {Dataset} />
                      <Route exact path="/Visualization" component = {Visualization} />
                      <Route exact path="/Performance" component = {Performance} />
                      <Route exact path="/AboutUs" component = {AboutUs} />
                      <Route exact path="/ContactUs" component = {ContactUs} />
                      <Route exact path="/FirstPrediction" component = {FirstPrediction} />
                      <Route exact path="/SecondPrediction" component = {SecondPrediction} />
                      <Route exact path="/ThirdPrediction" component = {ThirdPrediction} />
                      <Route exact path="/PredictionResult" component = {Result} />

                      <Route exact path="/DecisionTreeMainPage" component = {DecisionTreeMainPage} />                  


                      <Route   component= {NotFound} />
                    </Switch>

                  </Col>
              </Row>
          </Container>


      </Router>

    </div>


  );
}

export default App;
