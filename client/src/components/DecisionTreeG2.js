import React, { Component } from 'react'
// import Header from './/Header';
// import Footer from './Footer'
// import { Row,Col,Form,FormGroup,Label,Input,Button} from 'reactstrap';
// import {BrowserRouter as Router, Route} from 'react-router-dom'
import datamind from '../assets/images/DataMind.jpeg';
import Header from './/Header';
import Footer from './Footer'

import Tree from 'react-d3-tree';
// import './custom-tree.css';
import '../assets/style/custom-tree.css';

// This is a simplified example of an org chart with a depth of 2.
// Note how deeper levels are defined recursively via the `children` property.
const G2 = {
  name: 'G1',
  attributes: {
    G1: 'fail/pass?',
  },
  children: [
    {
      name: 'FAIL',
      attributes: {
        G1:'fail',
      },
    },
    {
      name: 'G1',
      attributes: {
        G1_:'pass',
        G1:'medium/high?',
      },
      children:[
        {
          name: 'HIGH',
          attributes: {
            G1:'high',
      },
        },
        {
          name: 'higherEdu',
          attributes: {
            G1:'medium',
            higherEdu:'yes/no?',
          },
          children:[
            {
              name: 'MEDIUM',
              attributes: {
                higherEdu:'yes',
              }
            },
            {
              name: 'failures',
              attributes: {
                higherEdu:'no',
                number:'<=1.5',
              },
              children:[
                {
                  name: 'FAIL',
                  attributes: {
                    failures:'more than 1.5',
                  },
                },
                {
                  name: 'MEDIUM',
                  attributes: {
                    failures:'less than 1.5',
                  },
                }
              ]
            }
          ]
        }
      ]
    }
  ],
}

// export default function OrgChartTree() {
//   return (
//     // `<Tree />` will fill width/height of its container; in this case `#treeWrapper`.
//     <div id="treeWrapper" style={{ width: '50em', height: '20em' }}>
//       <Tree data={orgChart} />
//     </div>
//   );
// }
export default function StyledNodesTree() {
    return (
      <div className="container">
        <Header/>
        <div className="card  text-white mb-5">
          <img src={datamind} className="img-fluid" alt="Responsive image" style={{"height":"450px","width":"100%"}}></img>
          <div className="card-img-overlay" style={{position:"absolte"}}>
              <h1 className="card-title text-center font-weight-bold" style={{"font-size":"49px"}} >Meet Our Dataset</h1>
          
          </div>
        </div>

        <div className=" mt-5 text-center h2">Fail/ Medium/ High 2nd exam</div>


        <div id="treeWrapper" style={{width: '1100px', height: '700px'}}>
          <Tree
            data={G2}
            translate={{ x: 50, y: 300 }}
            rootNodeClassName="node__root , mt-5"
            branchNodeClassName="node__branch"
            leafNodeClassName="node__leaf"
          />
        </div>
        <Footer/>
      </div>
    );
  }
