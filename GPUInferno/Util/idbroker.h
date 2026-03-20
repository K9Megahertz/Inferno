#pragma once




namespace Inferno {

	class IDBroker {

	public:


		static int GenID() {

			return nextid++;
		}


	private:

		static int nextid;

	};

}