import React, { Component } from 'react'


import Tree from 'react-d3-tree';
// import './custom-tree.css';
import '../assets/style/custom-tree.css';

// This is a simplified example of an org chart with a depth of 2.
// Note how deeper levels are defined recursively via the `children` property.
const G3 = {
  name: 'G2',
  attributes: {
    G2: 'high/ not high?',
  },
  children: [
    {
      name: 'HIGH',
      attributes: {
        G2:'high',
      },
    },
    {
      name: 'G2',
      attributes: {
        G2_:'not high',
        G2:'fail/medium?',
      },
      children:[
        {
          name: 'G1',
          attributes: {
            G2:'fail',
            G1:'high/not high',
      },
      children:[
        {
          name: 'HIGH',
          attributes: {
            G1:'high',
      },
        },
        {
          name: 'MEDIUM',
          attributes: {
            G1:'not high',
      },
    }
      ]
        },
        {
          name: 'G1',
          attributes: {
            G2:'fail',
            G1:'fail/medium?',
          },
          children:[
            {
              name: 'MEDIUM',
              attributes: {
                G1:'medium',
              }
            },
            {
              name: 'FAIL',
              attributes: {
                G1:'fail',
              },
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
export default function SecondTree() {
    return (


      <div id="treeWrapper" style={{width: '100%', height: '650px', border:'solid 1px grey'}} >

        <Tree 
          translate={{ x: 80, y: 250 }}
          data={G3}
          rootNodeClassName="node__root"
          branchNodeClassName="node__branch"
          leafNodeClassName="node__leaf"
        />
      </div>

    );
  }
