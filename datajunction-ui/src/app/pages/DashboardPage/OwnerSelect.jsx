import { useContext, useEffect, useState } from 'react';
import DJClientContext from '../../providers/djclient';
import Control from './FieldControl';

import Select from 'react-select';


export default function OwnerSelect() {
  const djClient = useContext(DJClientContext).DataJunctionAPI;
  const [retrieved, setRetrieved] = useState(false);
  const [users, setUsers] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const users = await djClient.users();
      setUsers(users);
      setRetrieved(true);
    };
    fetchData().catch(console.error);
  }, [djClient]);

  return (
    <span className="menu-link" style={{marginLeft: '30px', width: '400px'}}>
      {retrieved ?       <Select
      name="owner"
      isClearable
      label='Owner'
      components={{ Control }}
      defaultValue={{value: 'yshang@netflix.com', label: 'yshang@netflix.com (3152)'}}
      options={users?.map(user => {
        return {value: user.username, label: `${user.username} (${user.count})`};
      })}
      /> : ''}
    </span>
  );
};