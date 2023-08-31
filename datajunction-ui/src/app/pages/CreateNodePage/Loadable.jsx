/**
 * Asynchronously loads the component for the Node page
 */

import * as React from 'react';
import { lazyLoad } from '../../../utils/loadable';

export const CreateNodePage = props => {
  return lazyLoad(
    () => import('./index'),
    module => module.CreateNodePage,
    {
      fallback: <div></div>,
    },
  )(props);
};
