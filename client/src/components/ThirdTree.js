import React, { Component } from 'react'
// import Header from './/Header';
// import Footer from './Footer'
// import { Row,Col,Form,FormGroup,Label,Input,Button} from 'reactstrap';
// import {BrowserRouter as Router, Route} from 'react-router-dom'

import Tree from 'react-d3-tree';
// import './custom-tree.css';
import '../assets/style/custom-tree.css';

// This is a simplified example of an org chart with a depth of 2.
// Note how deeper levels are defined recursively via the `children` property.
const pf1 = {
  name: 'failures',
  attributes: {
    Number: '<=0.5?',
  },
  children: [
    {
      name: 'higher',
      attributes: {
        failures:'less than 0.5',
        higerEdu: 'yes/no?',
      },
      children:[
          {
            name: 'FAIL',
            attributes: {
              higerEdu: 'no',   
          },
        },
        {
            name: 'PASS',
            attributes: {
              higerEdu: 'yes',
            }
        }
      ]
    },
    {
    name: 'higher',
      attributes: {
        failures:' more than 0.5',
        higerEdu: 'yes/no?',
    },
    children: [
        {
            name: 'FAIL',
            attributes: {
              higerEdu:'no',
          }
        },
          {
              name: 'failures',
              attributes: {
                number:'<=2.5?',
                higerEdu:'yes',
              },
              children: [
                {
                    name: 'FAIL',
                    attributes: {
                        failures:'more than 2.5',
                  },
                },
                {
                    name: 'School',
                    attributes: {
                        failures:'less than 2.5',
                        school:'GP/MS?',
                  },
                  children:[
                      {
                        name: 'FAIL',
                        attributes: {
                            school:'MS',
                      }
                    },
                    {
                        name: 'failures',
                        attributes: {
                            school:'GP',
                            number:'<1.5?',
                      },
                      children:[
                        {
                          name: 'FAIL',
                          attributes: {
                            failures:'more than 1.5',
                        }
                      },
                      {
                        name: 'PASS',
                        attributes: {
                          failures:'less than 1.5',
                      }
                    },
                    ]
                    }
                  ],
                },
            ]
            },
           
    ]
          
}
  ],
};

const pf2 = {
  name: 'failures',
  attributes: {
    Number: '<=0.5?',
  },
  children: [
    {
      name: 'PASS',
      attributes: {
        failures:'less than 0.5',
      },
    },
    {
        name: 'FAIL',
        attributes: {
          failures:'more than 0.5',
        },
      }
    ]
};

const pf3 = {
  name: 'failures',
  attributes: {
    Number: '<=0.5?',
  },
  children: [
    {
      name: 'PASS',
      attributes: {
        failures:'less than 0.5',
      },
    },
    {
        name: 'school',
        attributes: {
          failures:'more than 0.5',
          school:'GP/MS?',
        },
        children:[
          {
            name:'FAIL',
            attributes:{
              school:'MS',
              higerEdu: 'yes/no?',
            },
            children:[
              {
                name: 'failures',
                attributes: {
                  higerEdu: 'yes',
                  Number: '<=1.5?',
                },
                children:[
                  {
                  name: 'failures',
                  attributes: {
                    failures: 'more than 1.5',
                    Number: '<=2.5?',
                },
                children:[
                  {
                    name: 'PASS',
                    attributes: {
                      failures: 'less than 2.5',
                  }
                  },
                  {
                    name: 'FAIL',
                    attributes: {
                      failures: 'more than 2.5',
                  }
                  }
                ]
              },{
                name: 'PASS',
                  attributes: {
                    failures: 'less than 1.5',
                  }
              }
                ]
              }
            ]
          },
          {
            name:'higherEdu',
            attributes:{
              school:'GP',

            }
          }
        ]
      }
    ]
};

// export default function OrgChartTree() {
//   return (
//     // `<Tree />` will fill width/height of its container; in this case `#treeWrapper`.
//     <div id="treeWrapper" style={{ width: '50em', height: '20em' }}>
//       <Tree data={orgChart} />
//     </div>
//   );
// }
export default function ThirdTree() {
    return (
      <div id="treeWrapper" style={{width: '100%', height: '650px', border:'solid 1px grey'}} >
        <Tree
          data={pf1}
          translate={{ x: 80, y: 250 }}
          rootNodeClassName="node__root"
          branchNodeClassName="node__branch"
          leafNodeClassName="node__leaf"
          zoom='0.7'
        />

        {/* <h4>Pass/ Fail 2nd exam</h4> */}
        {/* <Tree
          data={pf2}
          rootNodeClassName="node__root"
          branchNodeClassName="node__branch"
          leafNodeClassName="node__leaf"
        /> */}
        {/* <h4>Pass/ Fail final exam</h4> */}
        {/* <Tree
          data={pf3}
          rootNodeClassName="node__root"
          branchNodeClassName="node__branch"
          leafNodeClassName="node__leaf"
        /> */}
      </div>
    );
  }
