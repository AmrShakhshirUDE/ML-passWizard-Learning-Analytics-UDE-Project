import React, { Component }  from 'react'
import { Row,Card, CardTitle, CardText,Col } from 'reactstrap';
import datamind from '../assets/images/DataMind.jpeg';
import Header from './/Header';
import Footer from './Footer';
import FirstTree from './FirstTree';
import SecondTree from './SecondTree';
import ThirdTree from './ThirdTree';

export default class DecisionTreeMainPage extends Component {

    constructor(props) {
        super(props);
        this.state = {
          firstTree:false,
          secondTree:false,
          ThirdTree:true,
          
          firstTxt:false,
          secondTxt:false,
          ThirdTxt:true,

          firstTreeBox:'#91A3AA',
          secondTreeBox:'#91A3AA',
          ThirdTreeBox:'#545F63',

    
        };
      }
          


    render() {

        return (

      <div className="container">
        <Header/>
        <div className="card  text-white mb-5">
          <img src={datamind} className="img-fluid" alt="Responsive image" style={{"height":"300px","width":"100%"}}></img>
          <div className="card-img-overlay" style={{position:"absolte"}}>
              <h1 className="card-title text-center font-weight-bold" style={{"fontSize":"49px"}} >Decision Tree</h1>
          
          </div>
        </div>
        <Row className="justify-content-sm-start mt-1" >
          <Col md="2" >
          <Card body inverse   id="thirdtree"  style={{ height:'200px',textAlign:'center',backgroundColor:this.state.ThirdTreeBox,marginBottom:'25px'}}
            onClick={e => {e.preventDefault(); this.setState({firstTree:false,secondTree:false,ThirdTree:true,firstTreeBox:'#91A3AA',secondTreeBox:'#91A3AA',ThirdTreeBox:'#545F63 ',firstTxt:false,secondTxt:false,ThirdTxt:true})}}>
                <CardText></CardText>
                <CardTitle tag="h6" style={{"color":"white", paddingTop:'50%' , fontSize:"13pt"}}>
                First Exam 
                </CardTitle>
                {/* <Button>Button</Button> */}
            </Card>

            <Card body inverse  id="firsttree" style={{ height:'200px',textAlign:'center', backgroundColor:this.state.firstTreeBox , marginBottom:'25px'}} 
            onClick={e => {e.preventDefault(); this.setState({firstTree:true,secondTree:false,ThirdTree:false,firstTreeBox:'#545F63 ',secondTreeBox:'#91A3AA',ThirdTreeBox:'#91A3AA',firstTxt:true,secondTxt:false,ThirdTxt:false})}}>
                <CardText></CardText>
                <CardTitle tag="h6" style={{"color":"white", paddingTop:'50%' , fontSize:"13pt"}}>
                Second Exam
                  
                </CardTitle>
                {/* <Button>Button</Button> */}
            </Card>
            <Card body inverse  id="secondtree" style={{ height:'200px',textAlign:'center',backgroundColor:this.state.secondTreeBox,}}
            onClick={e => {e.preventDefault(); this.setState({firstTree:false,secondTree:true,ThirdTree:false,firstTreeBox:'#91A3AA',secondTreeBox:'#545F63 ',ThirdTreeBox:'#91A3AA',firstTxt:false,secondTxt:true,ThirdTxt:false})}}>
                <CardText></CardText>
                <CardTitle tag="h6" style={{"color":"white", paddingTop:'50%' , fontSize:"13pt"}}>
                  Final Exam 
                </CardTitle>
                {/* <Button>Button</Button> */}
            </Card>
           
          </Col>
          <Col md="10" >
              {
                  this.state.firstTree &&
                  <FirstTree/>
              }
              {
                  this.state.secondTree &&
                  <SecondTree/>
              }              
              {
                this.state.ThirdTree &&
                <ThirdTree/>
            }
          </Col>
        </Row>
        <Row className="justify-content-sm-center">
          <Col md="8" className="" >
          {
                  this.state.ThirdTxt &&
                  <p>Accuracy ~ 83.08%  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Precision: [fail pass] ~ [68.00% 86.67%]  </p>
                  
              }
              {
                  this.state.firstTxt &&
                  <p>Accuracy ~ 84.62%  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Precision: [fail medium high] ~ [78.13% 87.8% 81.25%]  </p>
              }              
              {
                this.state.secondTxt &&
                <p>Accuracy ~ 84.62%  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Precision: [fail medium high] ~ [66.67% 87.64% 94.12%]  </p>
            }
                    
          </Col>
        </Row>
        <Row className="justify-content-sm-start mt-3">
                  
                  <Col md="3" >
                      
                          <Card body inverse  style={{ height:'130px',textAlign:'center',backgroundColor:'#3e8e9e'}}>
                              <CardText></CardText>
                              <CardTitle tag="h6" style={{"color":"white"}}>
                              <strong style={{color:"white"}}>G1 :</strong> First Grade <br/>
                              <strong style={{color:"white"}}> G2 :</strong> Second Grade  <br/>
                              <p style={{fontSize:"10pt"}} >
                                (fail:less than 10 
                                / medium:10~14
                                / high:15~20)
                              </p>
                                                
                                </CardTitle>
                              {/* <Button>Button</Button> */}
                          </Card>
                  </Col> 
                  <Col md="3"  >
                          <Card body inverse  style={{ height:'130px',backgroundColor:'#e2975d' ,textAlign:'center'}}>
                              <CardText></CardText>
                              <CardTitle tag="h6" style={{"color":"white"}}>
                              <strong style={{color:"white"}}> failures :</strong> number of failures
                              </CardTitle>
                              {/* <Button color="secondary">Button</Button> */}
                          </Card>
                  </Col> 
                  <Col md="3" >
                          <Card body inverse  style={{ height:'130px',backgroundColor:'rgba(190,91,104)', textAlign:'center'}}>
                              <CardText></CardText>
                              <CardTitle tag="h6" style={{"color":"white"}}>
                              <strong style={{color:"white"}}> higherEdu : </strong>does the student want Higher Education (Yes/No)
                              </CardTitle>

                              {/* <Button color="secondary">Button</Button> */}
                          </Card>
                  </Col> 
                  <Col md="3">
                          <Card body inverse style={{ height:'130px',backgroundColor:'#75678C', textAlign:'center'}}>
                              <CardText></CardText>
                              <CardTitle tag="h6" style={{"color":"white"}}>
                              <strong style={{color:"white"}}>school :</strong> the School Name (GP/MS)
                              </CardTitle>

                              {/* <Button color="secondary">Button</Button> */}
                          </Card>
                  </Col> 
              </Row> 
              <br/><br/>
        
        <Footer/>
      </div>

        )
    }
}
