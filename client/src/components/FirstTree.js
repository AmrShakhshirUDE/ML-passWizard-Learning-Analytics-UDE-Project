import React from 'react';
import Tree from 'react-d3-tree';
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
export default function FirstTree() {
  return (


          <div id="treeWrapper" style={{width: '100%', height: '650px', border:'solid 1px grey'}} >
              <Tree
              data={G2}
              translate={{ x: 80, y: 250 }}
              rootNodeClassName="node__root , mt-5"
              branchNodeClassName="node__branch"
              leafNodeClassName="node__leaf"
            />
          </div>
  )
}
